import cohere
import os

cohere_api_key = os.getenv("CO_API_KEY")
co = cohere.Client(api_key=cohere_api_key)

message = """Can you fix the errors in the following python code: print('luke Wilsen)"""

response = co.chat(model="command-r-plus", message=message)

print(response.text)
