import os

from config import MILVUS_HOST, EMBEDDING_DIMENSION, MILVUS_COLLECTION_NAME
from utils import load_image_model_processor, encoder_image
from pymilvus import MilvusClient
from utils import load_text_model_tokenizer, encoder_text


image_model, image_processor = load_image_model_processor()
text_model, text_tokenizer = load_text_model_tokenizer()


class MyMilvusClient:
    def __init__(self):
        # Initialize Milvus client
        self.client = MilvusClient(MILVUS_HOST)

    def store_image_data(self, image_data_dir):
        image_paths = []
        for root, dirs, files in os.walk(image_data_dir):
            for file in files:
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

        image_embeddings = []
        for image_path in image_paths:
            image_embedding = encoder_image(image_path, image_model, image_processor)
            image_embeddings.append(image_embedding)

        # Create collection in Milvus
        self.client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            dimension=EMBEDDING_DIMENSION
        )

        # Prepare data for insertion
        entities = []
        for i, image_embedding in enumerate(image_embeddings):
            # Assuming the image_embedding is a vector of shape (EMBEDDING_DIMENSION,)
            entities.append({
                "id": i,
                "vector": image_embedding,
                "img_path": image_paths[i]
            })
        print(entities)

        # Insert data into Milvus
        insert_res = self.client.insert(collection_name=MILVUS_COLLECTION_NAME, data=entities)
        print(insert_res)

    def search_image(self, query, top_k=5):
        search_img_path = []
        query_vector = encoder_text(query, text_model, text_tokenizer)
        print(query_vector)
        res = self.client.search(
            collection_name=MILVUS_COLLECTION_NAME,  # target collection
            data=[query_vector],  # a list of one or more query vectors, supports batch
            limit=top_k,  # how many results to return (topK)
            output_fields=["img_path"],  # what fields to return
            anns_field="vector",  # which field to search
        )
        for d in res[0]:
            search_img_path.append(d['entity']['img_path'])
        return search_img_path


if __name__ == '__main__':
    milvus_client = MyMilvusClient()
    milvus_client.store_image_data("./images")




