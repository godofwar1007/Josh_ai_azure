import os
import asyncio
from qdrant_client import AsyncQdrantClient, models  # 👈 Import models here
from rank_bm25 import BM25Okapi

api = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MjBhZWUyOWYtMDU5OC00OGM0LWJkMDMtNGU4YTE3MGFiMDEzIn0.gJ9hIQCNvLleJydfDp0PrWVSZE5R7GAbi5cKI5f5GCM")
url = os.getenv("QDRANT_URL", "https://3463a70a-212e-4ee7-b027-a65de3c43055.us-east4-0.gcp.cloud.qdrant.io")

class RulesRetriever:
     def __init__(self):
          self.client = AsyncQdrantClient(
                url=url,
                api_key=api,
                cloud_inference=True
          )
          
     async def search(self, query: str, numberofchunks: int):
          # ✅ Use models.Document for text queries when cloud_inference=True
          # Replace "sentence-transformers/all-MiniLM-L6-v2" with your exact Qdrant Cloud model if different
          response = await self.client.query_points(
               collection_name="rules",
               query=models.Document(
                   text=query,
                   model="sentence-transformers/all-MiniLM-L6-v2" 
               ), 
               limit=numberofchunks * 4
          )
          
          results = response.points
          
          l = []
          k = []
          for r in results:
               p = r.payload
               m = p["metadata"]
               t = m.split()
               l.append(t)
               k.append(p["text"])
          
          bm25 = BM25Okapi(l)
          i = query.split()
          scores = bm25.get_scores(i)
          scores = sorted(enumerate(scores), reverse=True, key=lambda x: x[1])[:numberofchunks:1]
          
          selectedchunks = ""
          for idx, s in scores:
               selectedchunks = selectedchunks + (k[idx])
          return selectedchunks
     
async def main():
    retriever = RulesRetriever()
    query = "What are the eligibility criteria for JEE Advanced?"
    print(f"Searching for: {query}")
    result = await retriever.search(query, numberofchunks=2)
    print("\n--- Retrieved Chunks ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())