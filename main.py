from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
from groq import Groq  # Assuming you're using groq SDK or an API wrapper
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv()

# ENV
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
groq_api_key = os.environ["GROQ_API_KEY"]

# Models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

# Load the Snowflake embedding model
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-l-v2.0")
model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-l-v2.0")

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean pooling of the last hidden state as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

# Generate query vector using the correct model
query = "Hvornår har casa bailar åbent?"
query_vector = generate_embedding(query)

# Hybrid Search
collection = client.collections.get("FAQv4")

response = collection.query.hybrid(
    query=query,
    vector=query_vector,
    limit=10,
    alpha=0.3,
)

# Prepare context
retrieved_texts = [
    f"Category: {obj.properties['combined']}"
    for obj in response.objects
]
context = "\n".join(retrieved_texts)

# Compose prompt
prompt = f"""
Using the following context, answer the question:

Context:
{context}

Question:
{query}
please answer the question in danish.
"""

# Call Groq (you can also call it via requests.post if not using SDK)
groq_client = Groq(api_key=groq_api_key)
response = groq_client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

answer = response.choices[0].message.content
print("💬 Answer from Groq:")
print(answer)

client.close()
groq_client.close()