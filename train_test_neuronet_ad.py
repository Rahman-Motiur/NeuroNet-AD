
import os
import argparse
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from subject_level_split import (
    set_seed,
    load_subject_table,
    stratified_subject_holdout,
    expand_to_slice_level,
    load_and_join_metadata,
    make_cv_folds,
    MRISliceMetaDataset,
    DatasetConfig,
)
from models.neuronet_ad import NeuroNetAD
from utils.metrics import classification_report_from_logits


def get_transforms():
    return None


def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for batch in loader:
        imgs = batch["image"].to(device)
        meta = batch["meta"].to(device) if batch["meta"].numel() > 0 else None
        labels = batch["label"].to(device)

        logits = model(imgs, meta)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    total_loss /= len(loader.dataset)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = classification_report_from_logits(logits, labels)
    return total_loss, metrics


def train_fold(train_df, val_df, feature_cols: List[str], args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    cfg = DatasetConfig(image_size=(args.img_size, args.img_size), metadata_cols=feature_cols)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(train_df["label"].unique().tolist()))}

    train_ds = MRISliceMetaDataset(train_df, cfg, label_to_idx=label_to_idx, transforms=get_transforms())
    val_ds   = MRISliceMetaDataset(val_df,   cfg, label_to_idx=label_to_idx, transforms=get_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    meta_dim = len(cfg.metadata_cols) if cfg.metadata_cols is not None else 0
    model = NeuroNetAD(num_classes=len(label_to_idx), meta_dim=meta_dim,
                       img_token_mode=args.img_token_mode, mgca_heads=args.mgca_heads, mgca_embed=args.mgca_embed)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        scheduler.step()

        acc = val_metrics["accuracy"]
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_metrics['accuracy']:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_metrics['accuracy']:.4f}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping (patience {args.patience}) at epoch {epoch}.")
            break

    # load best
    model.load_state_dict(best_state)
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects_csv", type=str, required=True, help="CSV: subject_id,label,slice_dir")
    parser.add_argument("--metadata_csv", type=str, default=None, help="CSV with per-subject numeric features")
    parser.add_argument("--feature_cols", type=str, default="Age,MMSE,FAQ_Score,Global_CDR,Weight",
                        help="Comma-separated metadata column names to use")
    parser.add_argument("--out_dir", type=str, default="runs_neuronet_ad")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # MGCA params
    parser.add_argument("--img_token_mode", type=str, default="channels", choices=["channels", "spatial"])
    parser.add_argument("--mgca_heads", type=int, default=4)
    parser.add_argument("--mgca_embed", type=int, default=256)

    # protocol
    parser.add_argument("--no_cv", action="store_true", help="Skip 5-fold CV and only train a single fold")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    subjects_df = load_subject_table(args.subjects_csv)
    trainval_subjects_df, test_subjects_df = stratified_subject_holdout(subjects_df, test_size=0.20, seed=args.seed)

    trainval_slices_df = expand_to_slice_level(trainval_subjects_df)
    test_slices_df     = expand_to_slice_level(test_subjects_df)

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    trainval_slices_df, scaler = load_and_join_metadata(trainval_slices_df, args.metadata_csv, feature_cols)
    test_slices_df, _          = load_and_join_metadata(test_slices_df, args.metadata_csv, feature_cols, scaler=scaler)

    # 5-fold CV
    folds = make_cv_folds(trainval_slices_df, n_splits=5, seed=args.seed)
    fold_models = []
    fold_accs = []

    active_folds = [folds[0]] if args.no_cv else folds

    for fold_idx, (train_df, val_df) in enumerate(active_folds, start=1):
        print(f"\n==== Fold {fold_idx} ====")
        model, best_val_acc = train_fold(train_df, val_df, feature_cols, args)
        fold_models.append(model)
        fold_accs.append(best_val_acc)

    print("\nCross-val val accuracies:", fold_accs)
    print(f"Mean ± SD: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    # Evaluate on held-out TEST using the first/best fold model
    test_cfg = DatasetConfig(image_size=(args.img_size, args.img_size), metadata_cols=feature_cols)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(trainval_slices_df['label'].unique().tolist()))}
    test_ds = MRISliceMetaDataset(test_slices_df, test_cfg, label_to_idx=label_to_idx, transforms=None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    best_model = fold_models[int(np.argmax(fold_accs))]
    best_model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = run_epoch(best_model, test_loader, criterion, optimizer=None, device=device)
    print(f"\nHELD-OUT TEST | Loss {test_loss:.4f} | Acc {test_metrics['accuracy']:.4f} "
          f"| Prec {test_metrics['precision']:.4f} | Rec {test_metrics['recall']:.4f} | F1 {test_metrics['f1']:.4f}")

    # Save final best model
    torch.save(best_model.state_dict(), os.path.join(args.out_dir, "best_fold_model.pt"))


if __name__ == "__main__":
    main()
