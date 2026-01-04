import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT =", PROJECT_ROOT)
print("PROJECT_ROOT CONTENTS =", os.listdir(PROJECT_ROOT))

#sys.path.insert(0, os.path.abspath("./UniDepth"))
from unidepth.models import UniDepthV2


import torch
from torch.utils.data import DataLoader

from dataset import GTRSDataset
from losses import total_loss

# -------------------------------------------------
# Load pretrained UniDepthV2
# -------------------------------------------------
name = "unidepth-v2-vitl14"   # or unidepth-v2-vits14
model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

model = model.device()
model.train()

# -------------------------------------
# -----------
# Freeze backbone (CRITICAL)
# -------------------------------------------------
for p in model.backbone.parameters():
    p.requires_grad = False

# Optional: print trainable params (sanity check)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable}/{total}")

# -------------------------------------------------
# Optimizer
# -------------------------------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
dataset = GTRSDataset("data/curated")
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# -------------------------------------------------
# Training loop
# -------------------------------------------------
for epoch in range(10):
    running_loss = 0.0

    for batch in loader:
        batch = {
            k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        loss = total_loss(batch, model)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch}: loss={running_loss/len(loader):.4f}")

# -------------------------------------------------
# Save ONLY fine-tuned weights
# -------------------------------------------------
torch.save(model.state_dict(), "unidepthv2_gtrs.pt")
