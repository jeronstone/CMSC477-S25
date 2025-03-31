import os
import shutil
import random

def split_images(source_folder, folder_80, folder_20):
    # Create destination folders if they don't exist
    os.makedirs(folder_80, exist_ok=True)
    os.makedirs(folder_20, exist_ok=True)
    
    # Get list of image files
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(('png'))]

    # Shuffle the list randomly
    random.shuffle(images)
    
    # Split images into 80% and 20%
    split_index = int(0.8 * len(images))
    images_80 = images[:split_index]
    images_20 = images[split_index:]
    
    # Move images to respective folders
    for img in images_80:
        shutil.move(os.path.join(source_folder, img), os.path.join(folder_80, img))
    
    for img in images_20:
        shutil.move(os.path.join(source_folder, img), os.path.join(folder_20, img))
    
    print(f"Moved {len(images_80)} images to {folder_80}")
    print(f"Moved {len(images_20)} images to {folder_20}")

# Example usage
source_folder = r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\train_images\\"
folder_80 = r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\train_images\train\\"
folder_20 = r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\train_images\validation\\"

split_images(source_folder, folder_80, folder_20)