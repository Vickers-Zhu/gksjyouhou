from openai import OpenAI
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import ast


client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def execute_embedding(excel_file_path="assets/members.xlsx"):
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    df['profile'].fillna('No Profile', inplace=True)
    df['company'].fillna('No Company', inplace=True)
    df['title'].fillna('No Title', inplace=True)
    df["combined"] = (
        "Profile: " + df.profile.str.strip() + "Company: " + df.company.str.strip() + "Title: " + df.title.str.strip()
    )
    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    df.to_csv('output/embedded_members.csv', index=False)

class CollectionDB:
    def __init__(self, csv_file, collection_name):
        self.csv_file = csv_file
        self.collection_name = collection_name
        self.client = chromadb.Client()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key, model_name="text-embedding-ada-002"
        )
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

    def add_data(self):
        # Read data from CSV
        df = pd.read_csv(self.csv_file)

        # Initialize lists for embeddings, metadatas, and ids
        embeddings = []
        metadatas = []
        ids = []

        # Process each row of the DataFrame
        for index, row in df.iterrows():
            # # Assuming 'ada_embedding' is already in the correct format
            embeddings.append(ast.literal_eval(row['ada_embedding']))

            # # Create metadata dictionary
            metadata = {"name": row['name'], "info": row['combined']}
            metadatas.append(metadata)

            # # Use the DataFrame index as the ID
            ids.append(str(index))

        # Add data to the collection
        self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
