from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)  # Initialize the OpenAI embeddings model

result = embedding.embed_query("Delhi is the capital of India.")  # Generate embeddings for the query
print(str(result))  # Print the resulting embeddings