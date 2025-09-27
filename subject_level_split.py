

import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from PIL import Image



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_subject_table(subjects_csv: str) -> pd.DataFrame:
 
    df = pd.read_csv(subjects_csv)
    required = {"subject_id", "label", "slice_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"subjects_csv missing columns: {missing}")
    return df


def expand_to_slice_level(subjects_df: pd.DataFrame, patterns=(".png", ".jpg", ".jpeg")) -> pd.DataFrame:

    rows = []
    for _, r in subjects_df.iterrows():
        s_id = r["subject_id"]
        label = r["label"]
        sdir = r["slice_dir"]
        if not os.path.isdir(sdir):
            raise FileNotFoundError(f"slice_dir not found: {sdir}")
        files = []
        for ext in patterns:
            files.extend(glob.glob(os.path.join(sdir, f"*{ext}")))
        files = sorted(files)
        if len(files) == 0:
            raise ValueError(f"No slice images found for subject_id={s_id} at {sdir}")
        for f in files:
            rows.append({"subject_id": s_id, "label": label, "slice_path": f})
    return pd.DataFrame(rows)



def stratified_subject_holdout(subjects_df: pd.DataFrame,
                               test_size: float = 0.20,
                               seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # One row per subject
    sub_df = subjects_df[["subject_id", "label"]].drop_duplicates().reset_index(drop=True)

    train_subj, test_subj = train_test_split(
        sub_df["subject_id"].values,
        test_size=test_size,
        random_state=seed,
        stratify=sub_df["label"].values,
        shuffle=True
    )
    trainval_df = subjects_df[subjects_df["subject_id"].isin(train_subj)].copy()
    test_df = subjects_df[subjects_df["subject_id"].isin(test_subj)].copy()
    return trainval_df, test_df


def make_cv_folds(slice_df_trainval: pd.DataFrame,
                  n_splits: int = 5,
                  seed: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:

    y = slice_df_trainval["label"].values
    groups = slice_df_trainval["subject_id"].values

    # Stratified by y, grouped by subject_id
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in sgkf.split(slice_df_trainval, y=y, groups=groups):
        train_df = slice_df_trainval.iloc[tr_idx].reset_index(drop=True)
        val_df   = slice_df_trainval.iloc[va_idx].reset_index(drop=True)

        # Safety checks: no overlap of subjects between train and val
        train_subjects = set(train_df["subject_id"].unique())
        val_subjects   = set(val_df["subject_id"].unique())
        assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected between train and val!"
        folds.append((train_df, val_df))
    return folds


def load_and_join_metadata(slice_df: pd.DataFrame,
                           metadata_csv: Optional[str],
                           feature_cols: Optional[List[str]] = None,
                           scaler: Optional[StandardScaler] = None
                           ) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:

    if metadata_csv is None:
        return slice_df, None

    meta_df = pd.read_csv(metadata_csv)
    if "subject_id" not in meta_df.columns:
        raise ValueError("metadata.csv must contain 'subject_id'")

    # Auto-detect feature columns if not provided: all numeric except subject_id
    if feature_cols is None:
        feature_cols = [c for c in meta_df.columns if c != "subject_id" and np.issubdtype(meta_df[c].dtype, np.number)]
        # You can also specify a fixed list: ["Age", "MMSE", "FAQ_Score", "Global_CDR", "Weight"]

    # Scale numeric metadata for model stability
    if scaler is None:
        scaler = StandardScaler()
        meta_df[feature_cols] = scaler.fit_transform(meta_df[feature_cols])
    else:
        meta_df[feature_cols] = scaler.transform(meta_df[feature_cols])

    # Merge onto slice_df
    out = slice_df.merge(meta_df[["subject_id"] + feature_cols], on="subject_id", how="left")
    # If some subjects missing metadata, fill with zeros (or consider dropping)
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out, scaler



@dataclass
class DatasetConfig:
    image_size: Tuple[int, int] = (224, 224)  # resize (H, W)
    metadata_cols: Optional[List[str]] = None


class MRISliceMetaDataset(Dataset):
 
    def __init__(self,
                 df: pd.DataFrame,
                 cfg: DatasetConfig,
                 label_to_idx: Optional[Dict[str, int]] = None,
                 transforms=None):
        self.df = df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.transforms = transforms

        # Build label mapping if not provided
        if label_to_idx is None:
            labels = sorted(self.df["label"].unique().tolist())
            self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Choose metadata columns
        if cfg.metadata_cols is None:
            self.meta_cols = [c for c in self.df.columns
                              if c not in {"subject_id", "label", "slice_path"} and
                              np.issubdtype(self.df[c].dtype, np.number)]
        else:
            self.meta_cols = cfg.metadata_cols

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("L")  # MRI slices often single-channel; adjust if RGB
        if self.cfg.image_size is not None:
            img = img.resize(self.cfg.image_size, resample=Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row["slice_path"])
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            # default: convert to tensor [1, H, W] in [0,1]
            img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

        # Metadata vector (float32)
        if len(self.meta_cols) > 0:
            meta = torch.tensor(row[self.meta_cols].values, dtype=torch.float32)
        else:
            meta = torch.tensor([], dtype=torch.float32)

        y = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        sid = row["subject_id"]
        return {"image": img, "meta": meta, "label": y, "subject_id": sid}



if __name__ == "__main__":
    set_seed(42)

    # ---- Paths (edit) ----
    SUBJECTS_CSV = "subjects.csv"      # subject_id,label,slice_dir
    METADATA_CSV = "metadata.csv"      # optional per-subject numeric features


    subjects_df = load_subject_table(SUBJECTS_CSV)


    trainval_subjects_df, test_subjects_df = stratified_subject_holdout(subjects_df, test_size=0.20, seed=42)


    trainval_slices_df = expand_to_slice_level(trainval_subjects_df)
    test_slices_df     = expand_to_slice_level(test_subjects_df)


    feature_cols = ["Age", "MMSE", "FAQ_Score", "Global_CDR", "Weight"]  # edit to your feature names
    trainval_slices_df, scaler = load_and_join_metadata(trainval_slices_df, METADATA_CSV, feature_cols)
    test_slices_df, _          = load_and_join_metadata(test_slices_df, METADATA_CSV, feature_cols, scaler=scaler)


    folds = make_cv_folds(trainval_slices_df, n_splits=5, seed=42)


    cfg = DatasetConfig(image_size=(224, 224), metadata_cols=feature_cols)


    all_labels = sorted(trainval_slices_df["label"].unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    for fold_idx, (train_df, val_df) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold_idx} ---")
        print("Train subjects:", train_df['subject_id'].nunique(), "   slices:", len(train_df))
        print("Val subjects:  ", val_df['subject_id'].nunique(),   "   slices:", len(val_df))


        overlap = set(train_df.subject_id.unique()) & set(val_df.subject_id.unique())
        assert len(overlap) == 0, "Leakage detected!"

        train_ds = MRISliceMetaDataset(train_df, cfg, label_to_idx=label_to_idx, transforms=None)
        val_ds   = MRISliceMetaDataset(val_df,   cfg, label_to_idx=label_to_idx, transforms=None)


        sample = train_ds[0]
        print("Sample keys:", sample.keys(), " image:", sample["image"].shape, " meta:", sample["meta"].shape)

    test_ds = MRISliceMetaDataset(test_slices_df, cfg, label_to_idx=label_to_idx, transforms=None)
    print("\nHeld-out TEST subjects:", test_slices_df['subject_id'].nunique(), " slices:", len(test_slices_df))

