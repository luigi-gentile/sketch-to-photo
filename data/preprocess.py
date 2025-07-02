import os
import shutil
import random
import sys

# Paths to raw dataset folders containing original images and sketches
raw_images_dir = "raw/images"
raw_sketches_dir = "raw/raster_sketches"

# Output base directory where train/val/test splits will be saved
output_base_dir = "../data"

# Define dataset splits and their ratios
splits = ['train', 'val', 'test']
split_ratios = [0.8, 0.1, 0.1]

# Create necessary directories for each split and their subfolders ('photos' and 'sketches')
for split in splits:
    os.makedirs(os.path.join(output_base_dir, split, "photos"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, "sketches"), exist_ok=True)

# Gather all image file paths from the raw images directory along with their subfolder and filename
all_image_paths = []
for subfolder in os.listdir(raw_images_dir):
    folder_path = os.path.join(raw_images_dir, subfolder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            # Consider only image files with these extensions
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(folder_path, img_file)
                # Append a tuple: (subfolder name, image filename, full image path)
                all_image_paths.append((subfolder, img_file, full_path))

# Shuffle the list randomly to ensure random splitting
random.shuffle(all_image_paths)

# Calculate the total number of images and number of samples for each split
num_total = len(all_image_paths)
num_train = int(split_ratios[0] * num_total)
num_val = int(split_ratios[1] * num_total)
num_test = num_total - num_train - num_val  # Remaining images go to test split

def copy_files(items, split):
    """
    Copy image files and their corresponding sketches from the raw folders to
    the output directory, organized by split ('train', 'val', 'test').

    Also prints a progress percentage during copying.
    """
    total_files = len(items)
    for idx, (subfolder, img_file, img_path) in enumerate(items, 1):
        # Destination path for the real photo in the output split folder
        dest_photo = os.path.join(output_base_dir, split, "photos", img_file)
        shutil.copy(img_path, dest_photo)  # Copy the photo

        # Find corresponding sketch file path in raw sketches directory
        sketch_path = os.path.join(raw_sketches_dir, subfolder, img_file)
        if os.path.exists(sketch_path):
            # Destination path for the sketch in the output split folder
            dest_sketch = os.path.join(output_base_dir, split, "sketches", img_file)
            shutil.copy(sketch_path, dest_sketch)  # Copy the sketch
        else:
            # Warn if the corresponding sketch is missing
            print(f"\nWARNING: Sketch missing for {img_file} in subfolder {subfolder}")

        # Calculate current progress percentage
        percent = (idx / total_files) * 100
        # Print progress on the same line, overwriting previous output
        sys.stdout.write(f"\r[{split}] Progress: {percent:.2f}% ({idx}/{total_files})")
        sys.stdout.flush()

    print()  # Print newline when done with this split

# Copy images and sketches into train, validation, and test folders respectively
copy_files(all_image_paths[:num_train], "train")
copy_files(all_image_paths[num_train:num_train+num_val], "val")
copy_files(all_image_paths[num_train+num_val:], "test")

print("Dataset successfully split into train/val/test folders.")
