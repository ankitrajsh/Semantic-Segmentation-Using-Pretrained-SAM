import os
import numpy as np
import tifffile as tiff
from patchify import patchify
import cv2
from PIL import Image
import splitfolders
import random

# Setting constants
PATCH_SIZE = 256
USEFUL_LABEL_THRESHOLD = 0.05  # At least 5% of the patch must have useful labels

# Directories setup
# Directories setup
root_directory = r'G:\sernetdata\single'
img_dir = root_directory + r"\images"
mask_dir = root_directory + r"\masks"
output_img_dir = root_directory + r"\256_patches\images"
output_mask_dir = root_directory + r"\256_patches\masks"


# Function to process images and masks
def process_images_and_masks(img_path, mask_path):
    image = tiff.imread(img_path)
    mask = tiff.imread(mask_path)

    # Ensure image and mask are numpy arrays (handle any format discrepancies)
    image = np.array(image)
    mask = np.array(mask)

    # Crop to the nearest size divisible by PATCH_SIZE
    x_max = (image.shape[1] // PATCH_SIZE) * PATCH_SIZE
    y_max = (image.shape[0] // PATCH_SIZE) * PATCH_SIZE
    image = image[:y_max, :x_max]
    mask = mask[:y_max, :x_max]

    patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    patches_mask = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)

    saved_count = 0

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            img_patch = patches_img[i, j, 0]
            mask_patch = patches_mask[i, j, 0]

            # Check if the mask patch has enough useful data
            unique, counts = np.unique(mask_patch, return_counts=True)
            if np.sum(counts[1:]) / np.sum(counts) > USEFUL_LABEL_THRESHOLD:
                # Save image and mask patch if useful
                img_filename = f"{output_img_dir}patch_{i}_{j}.tif"
                mask_filename = f"{output_mask_dir}patch_{i}_{j}.tif"
                tiff.imwrite(img_filename, img_patch)
                tiff.imwrite(mask_filename, mask_patch)
                saved_count += 1

    return saved_count

# Loop through all files in the image and mask directories
for img_file, mask_file in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(mask_dir))):
    if img_file.endswith('.tif') and mask_file.endswith('.tif'):
        print(f"Processing {img_file} and {mask_file}")
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        saved_patches = process_images_and_masks(img_path, mask_path)
        print(f"Saved {saved_patches} useful patches from {img_file} and {mask_file}")

input_folder = root_directory + '/256_patches'
output_folder = root_directory + '/data_for_training_and_testing'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)



import os
import numpy as np
import tifffile as tiff
from patchify import patchify
import splitfolders

# Constants
PATCH_SIZE = 256
USEFUL_LABEL_THRESHOLD = 0.05

# Correcting paths using raw strings
root_directory = r'G:\sernetdata\single'
img_dir = os.path.join(root_directory, "images")
mask_dir = os.path.join(root_directory, "masks")
output_img_dir = os.path.join(root_directory, "256_patches", "images")
output_mask_dir = os.path.join(root_directory, "256_patches", "masks")

# Ensure directories exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Function to process and save patches
def process_images_and_masks(img_path, mask_path):
    image = tiff.imread(img_path)
    mask = tiff.imread(mask_path)
    image = image[:image.shape[0] // PATCH_SIZE * PATCH_SIZE, :image.shape[1] // PATCH_SIZE * PATCH_SIZE]
    mask = mask[:mask.shape[0] // PATCH_SIZE * PATCH_SIZE, :mask.shape[1] // PATCH_SIZE * PATCH_SIZE]

    patches_img = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    patches_mask = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            img_patch = patches_img[i, j, 0]
            mask_patch = patches_mask[i, j, 0]
            if np.sum(mask_patch != 0) / np.prod(mask_patch.shape) > USEFUL_LABEL_THRESHOLD:
                img_filename = os.path.join(output_img_dir, f"patch_{i}_{j}.tif")
                mask_filename = os.path.join(output_mask_dir, f"patch_{i}_{j}.tif")
                tiff.imwrite(img_filename, img_patch)
                tiff.imwrite(mask_filename, mask_patch)

# Processing files
for img_file in sorted(os.listdir(img_dir)):
    if img_file.endswith('.tif'):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file.replace('.tif', '_mask.tif'))
        if os.path.exists(mask_path):
            process_images_and_masks(img_path, mask_path)

# Splitting directories
input_folder = os.path.join(root_directory, "256_patches")
output_folder = os.path.join(root_directory, "data_for_training_and_testing")
if os.path.exists(input_folder):
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)
else:
    print(f"Input folder does not exist: {input_folder}")
