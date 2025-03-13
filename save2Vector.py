from datasets import load_dataset
import chromadb
from chromadb.utils.data_loaders import ImageLoader
import torch
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import os
if __name__ == "__main__":
    # img路径
    dataset_folder = '/mnt/d/myProjects/datasets/fashionpedia/img'

    # Instantiate the ChromaDB CLient
    chroma_client = chromadb.PersistentClient(path='/mnt/d/myProjects/datasets/multimodal_vdb/fashionpedia')
    # Instantiate the ChromaDB Image Loader
    image_loader = ImageLoader()
    # Instantiate CLIP embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLIP = OpenCLIPEmbeddingFunction(checkpoint="/mnt/d/myProjects/datasets/fashionpedia/open_clip_pytorch_model.bin",
                                     device=device)
    # 构建图像矢量数据库
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

    # Initialize lists for ids and uris (uniform resource identifiers, which in this case is just the path to the image)
    ids = []
    uris = []

    # Iterate over each file in the dataset folder
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith('.png'):
            file_path = os.path.join(dataset_folder, filename)

            # Append id and uri to respective lists
            ids.append(str(i))
            uris.append(file_path)

    # Assuming multimodal_db is already defined and available
    image_vdb.add(
        ids=ids,
        uris=uris
    )

    print("Images added to the database.")
