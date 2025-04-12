import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.transformer import DeformableTransformer

class RFDETR(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.transformer = DeformableTransformer()
        self.class_embed = nn.Linear(256, num_classes)
        self.bbox_embed = MLP(256, 256, 4, 3)

    def forward(self, x):
        features = self.backbone(x)['out']
        src = self.input_proj(features)
        hs = self.transformer(src)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
