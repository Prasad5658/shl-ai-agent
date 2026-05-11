import json

import requests


url = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

response = requests.get(url, timeout=30)
response.raise_for_status()

data = json.loads(response.text, strict=False)

with open("catalog/shl_catalog.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Catalog downloaded successfully!")
