from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
documents = [
    "Delhi is the capital of India.",
    "The Taj Mahal is located in Agra.",
    "Mumbai is the financial capital of India."
]

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)  # Initialize the OpenAI embeddings model

result = embedding.embed_documents(documents)  # Generate embeddings for the documents
print(str(result))  # Print the resulting embeddings