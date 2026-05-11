import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

with open("catalog/shl_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

documents = []
metadata = []

for item in catalog:
    name = item.get("name", "")
    description = item.get("description", "")
    url = item.get("link", "")
    test_type = ", ".join(item.get("keys", []))
    duration = item.get("duration", "")
    remote = item.get("remote", "")
    adaptive = item.get("adaptive", "")

    text = f"""
    Name: {name}
    Description: {description}
    Test Type: {test_type}
    Duration: {duration}
    Remote: {remote}
    Adaptive: {adaptive}
    """

    documents.append(text)
    metadata.append(
        {
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": description,
            "duration": duration,
            "remote": remote,
            "adaptive": adaptive,
        }
    )

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "embeddings/faiss_index.bin")

with open("embeddings/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index created!")
