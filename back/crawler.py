import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from mimetypes import guess_extension
import requests
from PIL import Image
from pillow_heif import register_heif_opener
from config_loader import config
import re
import imghdr


class BaseCrawler:
    def __init__(self, prefix, num_workers, verbose):
        register_heif_opener()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        self.prefix = prefix
        self.num_workers = num_workers
        self.verbose = verbose
        self.download_path = config["PATH"]["train"]
        os.makedirs(self.download_path, exist_ok=True)

    def crawling(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for page in range(1, self.pages + 1):
                executor.submit(self.run_crawling_thread, page)

    def run_crawling_thread(self, page):
        if self.verbose:
            print(f"start crawling, query={self.query}, page={page}")

        desks = self.fetch_desks(page)
        for desk in desks:
            self.download_image(
                desk["image_url"],
                desk["name"],
            )
    
    def fetch_desks(self, page):
        ...

    def sanitize_filename(self, filename):
        sanitized_filename = re.sub(r"[\/\\\:\*\?\"\<\>\|\s]", "_", filename)
        sanitized_filename = "".join(c for c in sanitized_filename if c.isprintable())
        return sanitized_filename

    def download_image(self, url, filename):
        response = requests.get(url)

        if response.status_code == 200:
            file_extension = guess_extension(response.headers.get("content-type"))
            image_data = response.content

            if file_extension is None:
                return

            image_data = self.validate_image_data(image_data)
            if not image_data:
                return

            file_extension = ".jpg"
            filename = self.prefix + "_" + filename + file_extension

            with open(os.path.join(self.download_path, filename), "wb") as f:
                f.write(image_data)

    def convert_heif_to_jpg(self, image_data):
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")

        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
    
    def validate_image_data(self, image_data):
        try:
            image = Image.open(BytesIO(image_data))
            pixels = image.load()
            image = image.convert("RGB")

            with BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_data = buffer.getvalue()
            
            return image_data
        except:
            return None


class DeskCrawler(BaseCrawler):
    def __init__(self, query, num_pages, num_workers=8, verbose=False):
        super().__init__("ohouse", num_workers, verbose)
        self.query = query
        self.pages = num_pages
        self.num_workers = num_workers
        self.verbose = verbose

    def fetch_desks(self, page):
        api_url = "https://ohou.se/cards/feed.json"
        params = {
            "v": 5,
            "query": self.query,
            "search_affect_type": "Typing",
            "page": page,
            "per": 48,
        }

        response = requests.get(api_url, params=params, headers=self.headers)
        if response.status_code != 200:
            raise Exception("Failed to fetch data")

        desks = []
        fetched_desks = response.json()["cards"]
        for fetched_desk in fetched_desks:
            desk = {
                "image_url": fetched_desk["image"]["url"],
                "name": self.sanitize_filename(str(fetched_desk["id"]))
            }
            desks.append(desk)
        return desks


if __name__ == "__main__":

    from config_loader import config

    train_path = config["PATH"]["train"]

    desk_crawler = DeskCrawler(query="데스크셋업", num_pages=52, num_workers=8, verbose=True)
    desk_crawler.crawling()
