import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator  # Import necessary for controlling colorbar ticks
from tqdm import tqdm



# Function to save the numpy array as an image with a colorbar
def save_image(data, filename, title):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('jet')  # Colormap
    norm = Normalize(vmin=data.min(), vmax=data.max())  # Normalize color range based on data
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis('off')
    colorbar = fig.colorbar(cax)
    colorbar.locator = MaxNLocator(integer=True)    # only Integers are used for ticks
    colorbar.update_ticks()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Function to process your numpy data
def process_data(file_path, output_folder):
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)

    # Access its content
    inst_map = data.item().get('inst_map')
    type_map = data.item().get('type_map')

    # Extract file info for naming
    base_name = os.path.basename(file_path).split('.')[0]  # e.g., '0_0'
    folder_num, image_index = base_name.split('_')

    # Create output directory for this folder number if it does not exist
    output_folder_path = os.path.join(output_directory_base, f'fold{folder_num}')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    # Save images
    unique_inst = np.unique(inst_map[inst_map != 0])  # Unique non-zero integers
    save_image(inst_map, os.path.join(output_folder_path, f"{base_name}_inst.png"),
               f"Inst Map: {len(unique_inst)}")
    save_image(type_map, os.path.join(output_folder_path, f"{base_name}_type.png"), "Type Map")

if __name__ == "__main__":

    # Specify the directory containing the .npy files
    # directory_path = 'path/to/your/folder'
    # output_directory_base = 'path/to/output/folder'

    directory_path = '/home/guoj5/Documents/pannuke_data_out/fold2/labels'
    output_directory_base = '/home/guoj5/Documents/pannuke_data_out/labels_parsed'

    # Ensure the output directory exists
    if not os.path.exists(output_directory_base):
        os.makedirs(output_directory_base)

    # Iterate over each file in the directory
    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith('.npy'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)

            # Process the data
            process_data(file_path, output_directory_base)
            print(f"Processed {filename}")
