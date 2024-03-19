import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from mimetypes import guess_extension
import requests
from PIL import Image
from pillow_heif import register_heif_opener
import re


class DeskCrawler:
    def __init__(
        self, base_download_path, queries, num_pages, num_workers=8, verbose=False
    ):
        register_heif_opener()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        self.base_download_path = base_download_path
        self.queries = queries
        self.pages = num_pages
        self.num_workers = num_workers
        self.verbose = verbose

        for query in self.queries:
            directory_path = os.path.join(base_download_path, query)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

    def crawling(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for query in self.queries:
                for page in range(1, self.pages + 1):
                    executor.submit(self.run_crawling_thread, query, page)

    def run_crawling_thread(self, query, page):
        if self.verbose:
            print(f"start crawling, query={query}, page={page}")

        download_path = os.path.join(self.base_download_path, query)
        cards = self.fetch_cards(query, page)
        for card in cards:
            self.download_image(
                card["image_url"], self.sanitize_filename(card["id"]), download_path
            )

    def fetch_cards(self, query, page):
        api_url = "https://ohou.se/cards/feed.json"
        params = {
            "v": 5,
            "query": query,
            "search_affect_type": "Typing",
            "page": page,
            "per": 48,
        }

        response = requests.get(api_url, params=params, headers=self.headers)
        if response.status_code != 200:
            raise Exception("Failed to fetch data")

        cards = []
        fetched_cards = response.json()["cards"]
        for fetched_card in fetched_cards:
            card = {
                "id": str(fetched_card["id"]),
                "image_url": fetched_card["image"]["url"],
            }
            cards.append(card)
        return cards

    def sanitize_filename(self, filename):
        sanitized_filename = re.sub(r"[\/\\\:\*\?\"\<\>\|\s]", "_", filename)
        sanitized_filename = "".join(c for c in sanitized_filename if c.isprintable())
        return sanitized_filename

    def download_image(self, url, filename, download_path):
        response = requests.get(url)

        if response.status_code == 200:
            file_extension = guess_extension(response.headers.get("content-type"))
            image_data = response.content

            if file_extension is None:
                return

            if file_extension == ".heif":
                image_data = self.convert_heif_to_jpg(image_data)
                file_extension = ".jpg"

            filename = filename + file_extension

            with open(os.path.join(download_path, filename), "wb") as f:
                f.write(image_data)

    def convert_heif_to_jpg(self, image_data):
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")

        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            return buffer.getvalue()


if __name__ == "__main__":
    queries = [
        "독서실책상",
        "컴퓨터책상",
        "일자형책상",
        "코너형책상",
        "h형책상",
    ]
    desk_crawler = DeskCrawler(
        "./images/train", queries=queries, num_pages=5, verbose=True
    )
    desk_crawler.crawling()
