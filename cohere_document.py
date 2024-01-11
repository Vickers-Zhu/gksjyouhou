import cohere
import os
import pandas as pd
import hnswlib
import json
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

co = cohere.Client(os.environ["COHERE_API_KEY"])


def read_excel(file_path):

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path, header=None, skiprows=1)
    df.fillna('', inplace=True)
    data = []

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        if index > 10:
            return
        # Accessing each column by its name
        name = row[0]
        company = row[1]
        title = row[2]
        profile = row[3]
        # Do something with the data, for example, print it
        data.append(
            {
                "name": name,
                "company": company,
                "title": title,
                "profile": profile,
            }
        )
    return data


class Documents:

    def __init__(self, file_path):
        # self.sources = sources
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load(file_path)
        self.embed()
        self.index()

    # def load(self) -> None:
    #     """
    #     Loads the documents from the sources and chunks the HTML content.
    #     """
    #     print("Loading documents...")

        # for source in self.sources:
        #     elements = partition_html(url=source["url"])
        #     chunks = chunk_by_title(elements)
        #     for chunk in chunks:
        #         self.docs.append(
        #             {
        #                 "title": source["title"],
        #                 "text": str(chunk),
        #                 "url": source["url"],
        #             }
        #         )

    def load(self, file_path) -> None:
        """Loads the documents from the sources and chunks the HTML content.

        Args:
            file_path (string): filepath
        """
        # Read the Excel file into a pandas DataFrame
        print("Loading documents...")

        df = pd.read_excel(file_path, header=None, skiprows=1)
        df.fillna('', inplace=True)

        # Iterate over the DataFrame rows
        for index, row in df.iterrows():
            # Accessing each column by its name
            name = row[0]
            company = row[1]
            title = row[2]
            profile = row[3]
            # Do something with the data, for example, print it
            self.docs.append(
                {
                    "name": name,
                    "company": company,
                    "title": title,
                    "profile": profile,
                    "text_for_ebedding": profile + ". Company: " + company + ". Title: " + title,
                }
            )

    def embed(self) -> None:
        """
        Embeds the documents using the Cohere API.
        """
        print("Embedding documents...")

        batch_size = 90
        self.docs_len = len(self.docs)

        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text_for_ebedding"]
                     for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        cuments for efficient retrieval.
        """
        print("Indexing documents...")

        self.index = hnswlib.Index(space="ip", dim=1024)
        self.index.init_index(max_elements=self.docs_len,
                              ef_construction=512, M=64)
        self.index.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.index.get_current_count()}")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the given query.

        Parameters:
        query (str): The query to retrieve documents for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved  documents, with 'title', 'snippet', and 'url' keys.
        """
        docs_retrieved = []
        query_emb = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings

        doc_ids = self.index.knn_query(query_emb, k=self.retrieve_top_k)[0][0]
        docs_to_rerank = []
        for doc_id in doc_ids:
            docs_to_rerank.append(self.docs[doc_id]["text_for_ebedding"])

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0",
        )

        doc_ids_reranked = []
        for result in rerank_results:
            doc_ids_reranked.append(doc_ids[result.index])

        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "name": self.docs[doc_id]["name"],
                    "company": self.docs[doc_id]["company"],
                    "title": self.docs[doc_id]["title"],
                    "profile": self.docs[doc_id]["profile"],
                }
            )

        return docs_retrieved
