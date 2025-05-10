import torch
from torchvision.models import efficientnet_b3
from PIL import Image
import io
from src.utils import img_transform

# Model configuration
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
model_path = "models/Eff_net_b3_01_brain_tumor.pth"

# Load model
def load_model():
    model = efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(in_features=1536, out_features=len(class_names))
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu")
        )
    )
    model.eval()
    return model

# Initialize model
model = load_model()

# Prediction function
def predict_tumor(image_bytes: bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Transform image
    transformed_image = img_transform(image).unsqueeze(0)
    
    # Perform inference
    with torch.inference_mode():
        preds = model(transformed_image)
        probs = torch.softmax(preds, dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
        class_label = class_names[label_idx]
        confidence = probs[0, label_idx].item()
    
    return class_label, confidence
