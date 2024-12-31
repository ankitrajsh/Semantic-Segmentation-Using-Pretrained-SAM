# %%
!pip install datasets

# %%
from datasets import Dataset

# %%
from PIL import Image

# %%
import os

# %%
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())

# %%
image_dir = 'G:/sernetdata/AGGC2022_train/Processed5/Images'
mask_dir = 'G:/sernetdata/AGGC2022_train/Processed5/Masks'
    # Path to the directory containing mask files

# Retrieve the list of image and mask filenames
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Sort the lists to ensure corresponding images and masks align (if filenames are matching)
image_files.sort()
mask_files.sort()

# Function to load images safely
def load_images(files, directory):
    images = []
    for file in files:
        with Image.open(os.path.join(directory, file)) as img:
            images.append(img.copy())  # Important to use copy() to keep the image after closing the file
    return images

# Load images and masks into a dictionary
dataset_dict = {
    "image": load_images(image_files, image_dir),
    "label": load_images(mask_files, mask_dir)
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)

# %%
dataset

# %%
!pip install matplotlib

# %%
import matplotlib.pyplot as plt  # Ensure this import is included
import random
import numpy as np
from datasets import Dataset
from PIL import Image
import os

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
# Choose a random index from the dataset
img_num = random.randint(0, len(image_files) - 1)  # Use the length of image_files or dataset['image']
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

# Plotting the images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[0].set_title("Image")

# Plot the second image (mask) on the right
axes[1].imshow(np.array(example_mask), cmap='gray')  # Convert PIL Image to NumPy array and assume it's grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

# %%
import numpy as np
from torch.utils.data import Dataset

class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = self.get_bounding_box(ground_truth_mask)

        # Even if bounding box is default (empty), still process and return structured data
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs


    def get_bounding_box(self, ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            # Return a default bounding box that indicates an empty or invalid box
            return [0, 0, 1, 1]  # Small, non-zero area at the origin
      

        # Compute bounding box coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add random perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return [x_min, y_min, x_max, y_max]

# %%
!pip install git+https://github.com/facebookresearch/segment-anything.git

# %%
!pip install -q git+https://github.com/huggingface/transformers.git

# %%
# Initialize the processor
from transformers import SamProcessor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# %%
train_dataset = SAMDataset(dataset=dataset, processor=processor)

# %%
example = train_dataset[0]

# Check if the example is None, which might be the case if the bounding box was empty
if example is None:
    print("The first item in the dataset is None (possibly an empty bounding box).")
else:
    # If the example is not None, proceed to print shapes of each component
    for k, v in example.items():
        print(f"{k}: {v.shape}")

# %%
# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

# %%
batch["ground_truth_mask"].shape

# %%
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# %%
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

# %%
!pip install -q monai

# %%
from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# %%
import torch
from tqdm import tqdm
from statistics import mean

# %%
# Define device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Define device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Transfer model to the chosen device
model.to(device)
model.train()

# Training loop
num_epochs = 1
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

# %%
torch.save(model.state_dict(), "G:\sernetdata\model_checkpoint.pth")


