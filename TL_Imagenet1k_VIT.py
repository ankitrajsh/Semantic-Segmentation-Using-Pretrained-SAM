import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from transformers import ViTModel
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Read and preprocess the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask



# Define input size
IMAGE_SIZE = (256, 256)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mask_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor()
])



# Paths to images and masks
image_dir = "C:/Users/Nishith/Downloads/New_train_Patches"
mask_dir = "G:/New_VOC_Label_Patches"

dataset = SegmentationDataset(image_dir, mask_dir, transform=transform, mask_transform=mask_transform)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


class ViTUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ViTUNet, self).__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x).last_hidden_state
        features = features.permute(0, 2, 1).reshape(features.shape[0], 768, 14, 14)  # Reshape output

        output = self.decoder(features)
        return output



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTUNet(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.long().to(device)  # Convert masks to long dtype for CrossEntropyLoss

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    print("Training complete!")
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


torch.save(model.state_dict(), "segmentation_vit_unet.pth")

# Load the model later
model.load_state_dict(torch.load("segmentation_vit_unet.pth"))
model.to(device)




def predict(model, image_path):
    model.eval()
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        output = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get predicted mask
    
    return output

# Example usage
predicted_mask = predict(model, "path_to_test_image.jpg")

# Visualize result
plt.imshow(predicted_mask, cmap="jet")
plt.axis("off")
plt.show()
