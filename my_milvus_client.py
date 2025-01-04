import os
from config import MILVUS_HOST, EMBEDDING_DIMENSION, MILVUS_COLLECTION_NAME
from utils import load_image_model_processor, encoder_image
from pymilvus import MilvusClient
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
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

        # Define schema
        field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        field_vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR)
        field_img_path = FieldSchema(name="img_path", dtype=DataType.STRING)

        # Define collection schema
        schema = CollectionSchema(fields=[field_id, field_vector, field_img_path],
                                  description="Image embeddings collection")

        # Create collection in Milvus
        self.client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            dimension=EMBEDDING_DIMENSION,
            schema=schema
        )

        # Prepare data for insertion
        entities = []
        for i, image_embedding in enumerate(image_embeddings):
            # Assuming the image_embedding is a vector of shape (EMBEDDING_DIMENSION,)
            entities.append({
                "vector": image_embedding,
                "img_path": image_paths[i]
            })

        # Insert data into Milvus
        self.client.insert(collection_name=MILVUS_COLLECTION_NAME, entities=entities)


    def search_image(self, query, top_k=5):
        query_vectors = encoder_text(query, text_model, text_tokenizer)
        res = self.client.search(
            collection_name=MILVUS_COLLECTION_NAME,  # target collection
            data=query_vectors,  # a list of one or more query vectors, supports batch
            limit=top_k,  # how many results to return (topK)
            output_fields=["image_path"],  # what fields to return
        )
        return res







