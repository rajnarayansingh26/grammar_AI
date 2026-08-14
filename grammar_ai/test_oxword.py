import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OXFORD_BASE_URL")
APP_ID = os.getenv("OXFORD_APP_ID")
APP_KEY = os.getenv("OXFORD_APP_KEY")

word = "apple"

url = f"{BASE_URL}/words/en-gb"

headers = {
    "app_id": APP_ID,
    "app_key": APP_KEY
}

params = {
    "q": word
}

response = requests.get(
    url,
    headers=headers,
    params=params
)

print("Status:", response.status_code)
print(response.text)