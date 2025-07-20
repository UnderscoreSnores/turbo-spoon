import requests
headers = {"X-API-Key": "WASDQE"}
r = requests.get("https://turbo-spoon-3o6f.onrender.com/predict?symbol=AAPL", headers=headers)
print(r.json())
