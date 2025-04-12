from models.rfdetr_model import RFDETR
from PIL import Image
import torchvision.transforms as T

model = RFDETR()
model.load_state_dict(torch.load("rfdetr_coco_weights.pth"))
model.eval()

image = Image.open("sample.jpg")
transform = T.Compose([
    T.Resize((800, 600)),
    T.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

# Post-processing step...
