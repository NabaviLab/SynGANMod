from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from torch.utils.data import Dataset

from .preprocessing import apply_clahe, binary_mask, read_grayscale
from .transforms import ToTensorNormalize


VIEW_TO_ID = {"CC": 0, "MLO": 1}
SIDE_TO_ID = {"Left": 0, "Right": 1}


class LongitudinalMammogramDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        image_size: int = 1024,
        transform: Optional[ToTensorNormalize] = None,
        apply_contrast: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.df = pd.read_csv(self.data_root / csv_path)
        self.image_size = image_size
        self.transform = transform or ToTensorNormalize()
        self.apply_contrast = apply_contrast

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, relative_path: str) -> str:
        return str(self.data_root / relative_path)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[index]

        prior = read_grayscale(self._resolve(row["prior_path"]), self.image_size)
        current = read_grayscale(self._resolve(row["current_path"]), self.image_size)
        breast_mask = binary_mask(self._resolve(row["breast_mask_path"]), self.image_size)

        if self.apply_contrast:
            prior = apply_clahe(prior)
            current = apply_clahe(current)

        tumor_mask_path = row.get("tumor_mask_path", "")
        has_tumor_mask = isinstance(tumor_mask_path, str) and tumor_mask_path.strip() != ""
        tumor_mask = binary_mask(self._resolve(tumor_mask_path), self.image_size) if has_tumor_mask else breast_mask * 0.0

        sample = {
            "case_id": str(row["case_id"]),
            "prior": prior,
            "current": current,
            "breast_mask": breast_mask,
            "tumor_mask": tumor_mask,
            "view_id": VIEW_TO_ID[str(row["view"])],
            "side_id": SIDE_TO_ID[str(row["side"])],
            "label": int(row["label"]),
            "has_tumor_mask": int(has_tumor_mask),
        }
        return self.transform(sample)
