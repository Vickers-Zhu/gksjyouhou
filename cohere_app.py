from cohere_chatbox import Chatbot
from cohere_document import Documents


class App:
    def __init__(self, chatbot: Chatbot):
        """
        Initializes an instance of the App class.

        Parameters:
        chatbot (Chatbot): An instance of the Chatbot class.

        """
        self.chatbot = chatbot

    def run(self):
        """
        Runs the chatbot application.
        """
        while True:
            # Get the user message
            message = input("User: ")

            # Typing "quit" ends the conversation
            if message.lower() == "quit":
                print("Ending chat.")
                break
            else:
                print(f"User: {message}")

              # Get the chatbot response
            response = self.chatbot.generate_response(message)

            # Print the chatbot response
            print("Chatbot:")
            flag = False
            for event in response:
                # Text
                if event.event_type == "text-generation":
                    print(event.text, end="")

                # Citations
                if event.event_type == "citation-generation":
                    if not flag:
                        print("\\n\\nCITATIONS:")
                        flag = True
                    print(event.citations)


# sources = [
#     {
#         "title": "Text Embeddings",
#         "url": "https://docs.cohere.com/docs/text-embeddings"},
#     {
#         "title": "Similarity Between Words and Sentences",
#         "url": "https://docs.cohere.com/docs/similarity-between-words-and-sentences"},
#     {
#         "title": "The Attention Mechanism",
#         "url": "https://docs.cohere.com/docs/the-attention-mechanism"},
#     {
#         "title": "Transformer Models",
#         "url": "https://docs.cohere.com/docs/transformer-models"}
# ]
documents = Documents("assets/members_en.xlsx")

chatbot = Chatbot(documents)

app = App(chatbot)

app.run()
