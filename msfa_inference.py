"""
MSFA-Net Inference
──────────────────
Configure the variables in the RUN CONFIGURATION section, then execute.

Modes:
  'image'  — predict a single MRI slice
  'folder' — predict all images in a directory, optionally save CSV
  'eval'   — full dataset evaluation with metrics
"""

import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score)

from msfa_net import (MSFANet, OASISMRIDataset, CLASS_NAMES, CDR_VALUES)


# ─────────────────────────────────────────────────────────────────────────────
# RUN CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT  = '/kaggle/working/msfa_best.pt'     # path to trained weights
MODE        = 'image'                             # 'image' | 'folder' | 'eval'
IMG_SIZE    = 128                                 # must match training size
BATCH_SIZE  = 64
NUM_WORKERS = 2                                   # DataLoader workers
DEVICE      = None                                # None = auto, or 'cuda'/'cpu'

# MODE = 'image'
IMAGE_PATH  = '/kaggle/input/imagesoasis/Data/Mild Dementia/OAS1_0028_MR1_mpr-1_100.jpg'

# MODE = 'folder'
FOLDER_PATH = '/kaggle/input/imagesoasis/Data/Mild Dementia/'
OUTPUT_CSV  = '/kaggle/working/predictions.csv'  # set to None to skip saving

# MODE = 'eval'
DATA_ROOT   = '/kaggle/input/imagesoasis/Data/'

# ─────────────────────────────────────────────────────────────────────────────


def get_inference_transform(img_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class MSFAInference:

    def __init__(self, checkpoint_path: str, img_size: int = 128,
                 device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device    = torch.device(device)
        self.img_size  = img_size
        self.transform = get_inference_transform(img_size)

        self.model = MSFANet(n_classes=4, img_size=img_size)
        state = torch.load(checkpoint_path, map_location=self.device,
                           weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded: {checkpoint_path}  |  Device: {self.device}")

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert('L')).unsqueeze(0).to(self.device)

    @staticmethod
    def _expected_cdr(probs: np.ndarray) -> float:
        return float(np.dot(probs, CDR_VALUES))

    # ── Single image ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_image(self, source: str | Path | Image.Image,
                      return_embedding: bool = False) -> dict:
        img = Image.open(source) if isinstance(source, (str, Path)) else source
        tensor = self._preprocess(img)

        if return_embedding:
            logits, emb = self.model(tensor, return_embedding=True)
            emb_list = emb.squeeze(0).cpu().numpy().tolist()
        else:
            logits   = self.model(tensor)
            emb_list = None

        probs      = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = int(probs.argmax())

        return {
            'predicted_class': pred_class,
            'predicted_label': CLASS_NAMES[pred_class],
            'confidence':      float(probs[pred_class]),
            'probabilities':   probs.tolist(),
            'expected_cdr':    self._expected_cdr(probs),
            'embedding':       emb_list,
        }

    # ── Batch folder ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_folder(self, folder: str | Path, batch_size: int = 64,
                       output_csv: str | None = None) -> list[dict]:
        folder = Path(folder)
        paths  = sorted(list(folder.glob('*.jpg')) +
                        list(folder.glob('*.jpeg')) +
                        list(folder.glob('*.png')))

        if not paths:
            print(f"[WARN] No images found in {folder}")
            return []

        results = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = [self.transform(Image.open(p).convert('L'))
                       for p in batch_paths]
            batch  = torch.stack(tensors).to(self.device)
            logits = self.model(batch)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

            for j, p in enumerate(batch_paths):
                pred = int(probs[j].argmax())
                results.append({
                    'filename':        p.name,
                    'predicted_class': pred,
                    'predicted_label': CLASS_NAMES[pred],
                    'confidence':      float(probs[j][pred]),
                    'prob_non_dem':    float(probs[j][0]),
                    'prob_very_mild':  float(probs[j][1]),
                    'prob_mild':       float(probs[j][2]),
                    'prob_moderate':   float(probs[j][3]),
                    'expected_cdr':    self._expected_cdr(probs[j]),
                })

            done = min(i + batch_size, len(paths))
            print(f"  {done}/{len(paths)} processed", end='\r')

        print(f"\nDone. {len(results)} images predicted.")

        if output_csv:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved → {output_csv}")

        return results

    # ── Full dataset evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_dataset(self, data_root: str, batch_size: int = 64,
                         num_workers: int = 2) -> dict:
        dataset = OASISMRIDataset(data_root,
                                   transform=get_inference_transform(self.img_size))
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

        all_preds, all_labels = [], []
        for imgs, labels in loader:
            imgs  = imgs.to(self.device, non_blocking=True)
            preds = self.model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        preds  = np.array(all_preds)
        labels = np.array(all_labels)
        acc    = float((preds == labels).mean())
        ba     = float(balanced_accuracy_score(labels, preds))
        cm     = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=CLASS_NAMES)

        per_class_acc = {
            CLASS_NAMES[c]: float((preds[labels == c] == c).mean())
            for c in range(len(CLASS_NAMES)) if (labels == c).sum() > 0
        }

        print(f"\nAccuracy        : {acc:.4f}")
        print(f"Balanced Acc    : {ba:.4f}")
        print("\nPer-class Accuracy:")
        for name, a in per_class_acc.items():
            print(f"  {name:<22}: {a:.4f}")
        print(f"\n{report}")
        print("Confusion Matrix:")
        print(cm)

        return {
            'accuracy':              acc,
            'balanced_accuracy':     ba,
            'per_class_accuracy':    per_class_acc,
            'classification_report': report,
            'confusion_matrix':      cm.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    inf = MSFAInference(CHECKPOINT, img_size=IMG_SIZE, device=DEVICE)

    if MODE == 'image':
        result = inf.predict_image(IMAGE_PATH)

        print(f"\nPrediction   : {result['predicted_label']}")
        print(f"Confidence   : {result['confidence']:.4f}")
        print(f"Expected CDR : {result['expected_cdr']:.3f}")
        print("\nProbabilities:")
        for name, p in zip(CLASS_NAMES, result['probabilities']):
            bar = '█' * int(p * 40)
            print(f"  {name:<22}: {p:.4f}  {bar}")

    elif MODE == 'folder':
        results = inf.predict_folder(FOLDER_PATH, batch_size=BATCH_SIZE,
                                     output_csv=OUTPUT_CSV)

        from collections import Counter
        counts = Counter(r['predicted_label'] for r in results)
        print("\nPrediction summary:")
        for label, n in counts.most_common():
            print(f"  {label:<22}: {n}")

    elif MODE == 'eval':
        metrics = inf.evaluate_dataset(DATA_ROOT, batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS)
