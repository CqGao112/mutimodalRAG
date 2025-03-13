import os
from datasets import load_dataset


# Function to save images
def save_images(dataset, dataset_folder, num_images=1000):
    for i in range(num_images):
        # Get the image data
        image = dataset['train'][i]['image']

        # Save the image
        image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))



# Define the folder to save the dataset images images
dataset_folder = '/mnt/d/myProjects/datasets/fashionpedia/img'
os.makedirs(dataset_folder, exist_ok=True)

# Load fashionpedia dataset available through HuggingFace
dataset = load_dataset("/mnt/d/myProjects/datasets/fashionpedia")

# Save the first 1000 images
save_images(dataset, dataset_folder, num_images=1000)