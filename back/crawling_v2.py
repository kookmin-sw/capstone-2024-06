import requests
import re
import os
from detect import count_class
from tqdm import tqdm


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_cards(query, page=1):
    api_url = "https://ohou.se/cards/feed.json"
    params = {
        "v": 5,
        "query": query,
        "search_affect_type": "Typing",
        "per": 48
    }

    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch data")
    
    cards = []
    fetched_cards = response.json()["cards"]
    for fetched_card in fetched_cards:
        card = {
            "id": str(fetched_card["id"]),
            "image_url": fetched_card["image"]["url"]
        }
        cards.append(card)
    return cards


def sanitize_filename(filename):
    sanitized_filename = re.sub(r"[\/\\\:\*\?\"\<\>\|]", "_", filename)
    sanitized_filename = "".join(c for c in sanitized_filename if c.isprintable())
    return sanitized_filename.replace(" ", "_")


def download_image(url, file_name, file_extension, download_folder):
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{download_folder}/{file_name}.{file_extension}", "wb") as f:
            f.write(response.content)

def Process_image_by_number_of_objects(data_dir):

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        # 해당 폴더가 디렉터리인지 확인
        if os.path.isdir(folder_path):
            print("Processing folder:", folder_name)

            # 각 이미지 파일에 대해 클래스별 객체 수 확인
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # 이미지 파일인지 확인
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    class_counts = count_class(file_path)
                    # 클래스별 객체 수로 이미지 처리
                    if class_counts <= cutline:
                        print("Deleting image:", file_name)
                        os.remove(file_path)


if __name__ == "__main__":
    base_download_folder = "./train_image"

    queries = [
        "독서실책상",
        "컴퓨터책상",
        "일자형책상",
        "코너형책상",
        "h형책상",
    ]  # 책상 종류

    # queries = ["독서실책상"]

    for query in queries:
        download_folder = f"{base_download_folder}/{query}"
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        for page in tqdm(range(1, 15), desc=f"Processing {query}"):  # 페이지 수 조절
            cards = get_cards(query, page=page)
            for card in cards:
                image_url = card["image_url"]
                file_extension = os.path.splitext(image_url)[1]
                download_image(
                    image_url,
                    sanitize_filename(card["id"]),
                    file_extension,
                    download_folder
                )

    # data_dir = "./train_image"
    # Process_image_by_number_of_objects(data_dir)