import requests
import json

url = "http://localhost:8000/prediction"

data = [[893,False,47.0,1,0,7.0,0,0]]
# Survived = 1
j_data = json.dumps(data)
headers = {'content-type':'application/json','Accept-Charset':'UTF-8'}
r = requests.post(url, data=j_data,headers=headers)
print(r,r.text)