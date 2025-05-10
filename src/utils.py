from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Image transformation pipeline
img_transform = transforms.Compose([
    transforms.Resize(320, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])