import chromadb
from chromadb.utils import embedding_functions
import openai
import os
import pandas as pd
from embedding import CollectionDB

# Initialize ChromaDB client
client = chromadb.Client()
client.heartbeat()

class Retriever:
    def __init__(self, collection_name="talent_profiles"):
        self.collection_name = collection_name
    
    def get_retrieval_results(self, input, k):
        cdb = CollectionDB("output/embedded_members.csv", self.collection_name)
        cdb.add_data()
        # print(cdb.collection.peek())
        # Use the combined profile data for the query
        retrieval_results = cdb.collection.query(
            query_texts=[input],
            n_results=k,
        )
        print(retrieval_results["metadatas"][0])
        print('-----------------')
        return retrieval_results["metadatas"][0]

class Generator:
    def __init__(self, openai_model="gpt-3.5-turbo"):
        self.openai_model = openai_model
        self.prompt_template_profile = """{profile}"""
        self.prompt_template_ques = """You're a helpful assistant. Based on the profiles and user's question, provide a recommendation from the aboving candidate profiles in Japanese:
            Question: {user_question}"""
        self.prompt_template_assistant = """"""

    def generate_response(self, retrieval_results, questions):
        prompts = []
        for result in retrieval_results:
            prompt = self.prompt_template_profile.format(profile=result.get('info'))
            prompts.append(prompt)
        prompts.reverse()
        messages=[{"role": "system", "content": prompt} for prompt in prompts]
        messages.append({"role": "user", "content": self.prompt_template_ques.format(user_question=questions)})
        print(messages)
        response = openai.OpenAI().chat.completions.create(
            model=self.openai_model,
            messages=messages
        )

        return response.choices[0].message.content

class Chatbot:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def answer(self, input):
        k = 5  # Number of profiles to retrieve
        retrieval_results = self.retriever.get_retrieval_results(input, k)
        return self.generator.generate_response(retrieval_results, input)

# Creating an instance of the Chatbot class
chatbot = Chatbot()

user_input = input("User prompt: ")
response = chatbot.answer(user_input)
print(f"Chatbot: {response}")
