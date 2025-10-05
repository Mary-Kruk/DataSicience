import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json

def classify_image(image_path, model_path, class_map_path):
    device = torch.device("cpu")
    try:
        with open(class_map_path, 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print(f"Error: Class names file '{class_map_path}' not found.")
        return None

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file '{model_path}' not found.")
        return None
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image '{image_path}' not found.")
        return None
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        return class_names[preds[0]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an animal image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, default="animal_classifier.pth", help="Path to model weights.")
    parser.add_argument("--class_map_path", type=str, default="class_names.json", help="Path to class names mapping.")
    args = parser.parse_args()
    prediction = classify_image(args.image_path, args.model_path, args.class_map_path)
    if prediction:
        print(f"Predicted animal: {prediction}")
