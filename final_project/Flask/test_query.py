import requests
import json

url = "http://localhost:8000/query"

query = '''SELECT * FROM titanic LIMIT 5;'''

j_query = json.dumps(query)

r = requests.post(url, query = query)

print(r, r.text)