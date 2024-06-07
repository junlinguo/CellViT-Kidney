import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

target_height = 500
def concatenate_images(image_paths, output_path, target_height=500):
    # Open images and resize them to the target height while maintaining aspect ratio
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        # Calculate the new width maintaining the aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * target_height)
        # Resize image
        img = img.resize((new_width, target_height), Image.ANTIALIAS)
        images.append(img)

    # Determine the total width
    total_width = sum(img.width for img in images)

    # Create a new image with the correct size
    new_im = Image.new('RGB', (total_width, target_height))

    # Paste each image into the new image
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    # Save the new image
    new_im.save(output_path)



# Paths to the folders
original_folder = '/home/guoj5/Documents/pannuke_data_out/fold2/images'         #'path/to/originals'
instance_folder = '/home/guoj5/Documents/pannuke_data_out/labels_parsed/fold2'  # path/to/instances'
type_folder = instance_folder
output_folder = '/home/guoj5/Documents/pannuke_data_out/labels_parsed/folder2_concat'  #'path/to/output'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each file in the original folder
for filename in tqdm(os.listdir(original_folder)):
    if filename.endswith('.png') and not filename.endswith('_inst.png') and not filename.endswith('_type.png'):
        base_name = os.path.splitext(filename)[0]

        # Construct file paths
        original_path = os.path.join(original_folder, filename)
        instance_path = os.path.join(instance_folder, f"{base_name}_inst.png")
        type_path = os.path.join(type_folder, f"{base_name}_type.png")

        # Define output path
        output_path = os.path.join(output_folder, f"{base_name}_concat.png")

        # Check if all files exist before processing
        if os.path.exists(original_path) and os.path.exists(instance_path) and os.path.exists(type_path):
            concatenate_images([original_path, instance_path, type_path], output_path)
            print(f"Concatenated image saved as {output_path}")
        else:
            print(f"Missing files for {base_name}, skipping...")
