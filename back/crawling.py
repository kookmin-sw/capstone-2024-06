import os
import requests

download_folder = os.path.dirname(os.path.abspath(__file__)) + "/image"
os.makedirs(download_folder, exist_ok=True)

api_url = "https://ohou.se/store/category.json"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
params = {
    "v": 2,
    "category_id": 10150004,
    "page": 0,
    "per": 24
}

response = requests.get(api_url, params=params, headers=headers)
if response.status_code == 200:
    products = response.json()["productions"]

    for product in products:
        desk_name = product["name"]
        desk_name = desk_name.replace(" ", "_").replace(".", "_").replace("/", "_")
        image_url = product["original_image_url"]
        image_response = requests.get(product["original_image_url"])

        if image_response.status_code == 200:
            with open(f"{download_folder}/{desk_name}.jpg", "wb") as file:
                file.write(image_response.content)