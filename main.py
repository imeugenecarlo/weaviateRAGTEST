from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
from groq import Groq  # Assuming you're using groq SDK or an API wrapper
import os
from dotenv import load_dotenv

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

# Hybrid Search
query = "What are the capitals of countries in Europe?"
query_vector = embedding_model.encode(query).tolist()
collection = client.collections.get("ChatMessage")

response = collection.query.hybrid(
    query=query,
    vector=query_vector,
    limit=5,
    alpha=0.5,
)

# Prepare context
retrieved_texts = [obj.properties["text"] for obj in response.objects]
context = "\n".join(retrieved_texts)

# Compose prompt
prompt = f"""
Using the following context, answer the question:

Context:
{context}

Question:
{query}
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