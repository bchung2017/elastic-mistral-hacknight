"""Check Elasticsearch, Mistral embeddings, and the squirrel corpus size.

pip install elasticsearch mistralai numpy requests
export ELASTIC_ENDPOINT=... ELASTIC_API_KEY=... MISTRAL_API_KEY=...
python smoke_test.py
"""
import os

import numpy as np
import requests
from elasticsearch import Elasticsearch
from mistralai.client import Mistral  # per the repo's mistral_guide.md

es = Elasticsearch(os.environ["ELASTIC_ENDPOINT"], api_key=os.environ["ELASTIC_API_KEY"])
print("elasticsearch", es.info()["version"]["number"])

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
resp = client.embeddings.create(model="mistral-embed", inputs=["squirrel eating a bagel", "hawk on a lamppost"])
v = np.array([d.embedding for d in resp.data])
print("mistral-embed dims", v.shape[1], "| norms", np.linalg.norm(v, axis=1).round(4), "| cos", round(float(v[0] @ v[1] / np.linalg.norm(v[0]) / np.linalg.norm(v[1])), 3))

url = "https://data.cityofnewyork.us/resource/gfqj-f768.json"
n = requests.get(url, params={"$select": "count(*)", "$where": "note_squirrel_park_stories IS NOT NULL"}, timeout=60).json()
print("squirrel stories with a note:", n)
