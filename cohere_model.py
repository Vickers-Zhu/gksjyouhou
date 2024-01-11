import cohere
import os

apiKey = os.environ.get('COHERE_API_KEY')

co = cohere.Client(apiKey)

prediction = co.chat(message='Howdy! 🤠 Give me a story', model='command')

# # print the predicted text
print(f'Chatbot: {prediction.text}')
