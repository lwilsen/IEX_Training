import requests

url = "http://localhost:8000/prediction"

r = requests.get(url)
print(r, r.text)