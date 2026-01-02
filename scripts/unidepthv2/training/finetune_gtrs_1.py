# training/finetune_gtrs.py
import torch
from torch.utils.data import DataLoader
#from unidepth.models.unidepthv2 import UniDepthV2
from dataset import GTRSDataset
from losses import total_loss
from unidepth.models import UniDepthV2

name = 'unidepth-v2-vitl14' # unidepth-v2-vits14, unidepth-v2-vits14
model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
#model = UniDepthV2.from_pretrained("unidepthv2")
model.cuda()
model.train()

# Freeze backbone (VERY IMPORTANT)
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

dataset = GTRSDataset("data/curated")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(10):
    for batch in loader:
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        loss = total_loss(batch, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")

torch.save(model.state_dict(), "unidepthv2_gtrs.pt")
