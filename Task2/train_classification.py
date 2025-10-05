import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import json
import os

def train_image_classifier(data_dir, num_epochs, lr, model_save_path, class_map_path):
    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset from the specified directory
    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=2)

    # Get the class names from the dataset folders and save them to a JSON file
    class_names = image_dataset.classes
    with open(class_map_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Found classes ({len(class_names)}): {class_names}")

    # Load a pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all the parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer to match the number of classes in our dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters of the new final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # Start the training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        model.train() # Set the model to training mode
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item() * inputs.size(0)

        # Calculate and print the loss for the epoch
        epoch_loss = running_loss / len(image_dataset)
        print(f'Loss: {epoch_loss:.4f}')

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to '{model_save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--data_dir", type=str, default="data/animals", help="Directory with image data.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--model_save_path", type=str, default="animal_classifier.pth", help="Path to save the model.")
    parser.add_argument("--class_map_path", type=str, default="class_names.json", help="Path to save class names.")
    args = parser.parse_args()
    
    train_image_classifier(args.data_dir, args.num_epochs, args.lr, args.model_save_path, args.class_map_path)