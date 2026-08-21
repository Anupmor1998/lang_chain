from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Initialize the HuggingFace embeddings model

text = "Delhi is the capital of India."
documents = [
    "Delhi is the capital of India.",
    "The Taj Mahal is located in Agra.",
    "Mumbai is the financial capital of India."]

# result  = embedding.embed_query(text) 
result  = embedding.embed_documents(documents)  # Generate embeddings for the documents
print(str(result))  # Print the resulting embeddings