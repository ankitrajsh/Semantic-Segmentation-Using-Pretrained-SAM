import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            with Image.open(image_path) as img:
                image = img.copy()
            with Image.open(mask_path) as mask:
                ground_truth_mask = np.array(mask.copy())
        except Exception as e:
            # In case of any exception, return a default item
            return None

        prompt = self.get_bounding_box(ground_truth_mask)
        if prompt == [0, 0, 1, 1]:  # This checks for your default "empty" bounding box
            return None  # Or consider providing a default non-empty bounding box

        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

    def get_bounding_box(self, ground_truth_map):
        # your existing bounding box calculation
        ...

# Loading the dataset
image_dir = 'G:/sernetdata/AGGC2022_train/Processed5/Images'
mask_dir = 'G:/sernetdata/AGGC2022_train/Processed5/Masks'
processor = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])
 # This needs to be defined or passed
train_dataset = SAMDataset(image_dir, mask_dir, processor)

# Creating the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True, collate_fn=lambda x: x)

# Usage of DataLoader
for batch in train_dataloader:
    if batch:  # Check if the batch is not empty
        print("Batch loaded successfully.")
        ground_truth_masks = [item["ground_truth_mask"] for item in batch if item is not None]
        if ground_truth_masks:  # Ensure there is at least one valid mask
            print("Ground truth mask shape:", ground_truth_masks[0].shape)
        else:
            print("No valid masks in this batch.")
    else:
        print("Empty batch, possibly due to filtering out invalid data.")


# Initialize the processor
from transformers import SamProcessor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


example = train_dataset[0]

# Check if the example is None, which might be the case if the bounding box was empty
if example is None:
    print("The first item in the dataset is None (possibly an empty bounding box).")
else:
    # If the example is not None, proceed to print shapes of each component
    for k, v in example.items():
        print(f"{k}: {v.shape}")

# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=False)


batch["ground_truth_mask"].shape


from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")


for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


import torch
from tqdm import tqdm
from statistics import mean

# Define device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



# Define device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Transfer model to the chosen device
model.to(device)
model.train()

# Training loop
num_epochs = 4
for epoch in range(num_epochs):
    epoch_losses = []  # to store loss values
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

    for batch in progress_bar:
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss for this batch and update tqdm description
        epoch_losses.append(loss.item())
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.4f}")

    # After all batches, print average loss for the epoch
    average_loss = mean(epoch_losses)
    print(f'Epoch {epoch+1}/{num_epochs}, Mean Loss: {average_loss:.4f}')