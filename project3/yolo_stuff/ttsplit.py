import os
import shutil
import random

def split_images(source_folder, folder_80, folder_20):
    # Create destination folders if they don't exist
    os.makedirs(folder_80, exist_ok=True)
    os.makedirs(folder_20, exist_ok=True)
    
    # Get list of image files
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(('jpg'))]

    # Shuffle the list randomly
    random.shuffle(images)
    
    # Split images into 80% and 20%
    split_index = int(0.8 * len(images))
    images_80 = images[:split_index]
    images_20 = images[split_index:]
    
    # Move images to respective folders
    for img in images_80:
        #print(img)
        #print(int(img.split('_')[1].split('.')[0]))
        idx = int(img.split('_')[1].split('.')[0])
        
        try:
            shutil.move(os.path.join(source_folder, img), os.path.join(folder_80, img))
            shutil.move(os.path.join(source_folder, f'frame_{idx}.txt'), os.path.join(folder_80, f'frame_{idx}.txt'))
        except:
            print(f'train: failed at {idx}')
    
    for img in images_20:
        #print(img)
        #print(int(img.split('_')[1].split('.')[0]))
        idx = int(img.split('_')[1].split('.')[0])
        
        try:
            shutil.move(os.path.join(source_folder, img), os.path.join(folder_20, img))
            shutil.move(os.path.join(source_folder, f'frame_{idx}.txt'), os.path.join(folder_20, f'frame_{idx}.txt'))
        except:
            print(f'validate: failed at {idx}')

    print(f"Moved {len(images_80)} images to {folder_80}")
    print(f"Moved {len(images_20)} images to {folder_20}")

# Example usage
source_folder = r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\train_images_f\\"
folder_80 = r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\train_images_p3f\train\\"
folder_20 = r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\train_images_p3f\validation\\"

split_images(source_folder, folder_80, folder_20)