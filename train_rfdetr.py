import torch
from models.rfdetr_model import RFDETR
from datasets import build_coco_dataset
from utils.matcher import HungarianMatcher
from utils.loss import SetCriterion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RFDETR(num_classes=91).to(device)

dataset_train = build_coco_dataset('train2017')
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True)

matcher = HungarianMatcher()
criterion = SetCriterion(num_classes=91, matcher=matcher).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
