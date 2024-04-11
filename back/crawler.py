import os
import re
import requests
import queue
import threading
from io import BytesIO

from mimetypes import guess_extension
from PIL import Image
from pillow_heif import register_heif_opener

from config_loader import config


class BaseCrawler:
    def __init__(self, path, prefix, num_workers, verbose):
        register_heif_opener()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.finished = False
        self.num_progressed = 0
        self.queue = queue.Queue()
        self.lock = threading.Lock()

        self.prefix = prefix
        self.num_workers = num_workers
        self.verbose = verbose
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def crawling(self):
        self.num_progressed = 0
        
        producer_thread = threading.Thread(target=self.producer)
        consumer_threads = []
        for _ in range(self.num_workers):
            consumer_threads.append(threading.Thread(target=self.consumer))
        
        producer_thread.start()
        for consumer_thread in consumer_threads:
            consumer_thread.start()
        
        producer_thread.join()
        for consumer_thread in consumer_threads:
            consumer_thread.join()
        
        if self.verbose:
            print(f"total {self.num_progressed} images downloaded")
    
    def producer(self):
        ...
    
    def consumer(self):
        while True:
            self.lock.acquire()
            if not self.queue.empty():
                image = self.queue.get()
                self.lock.release()
                self.download_image(image)

                if self.verbose:
                    with self.lock:
                        self.num_progressed += 1
                        print(f"Progressed {self.num_progressed} images", end="\r")
            else:
                self.lock.release()
                
                if self.finished:
                    break

    def download_image(self, image):
        url = image["url"]
        filename = self.sanitize_filename(image["name"])

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

            with open(os.path.join(self.path, filename), "wb") as f:
                f.write(image_data)
    
    def sanitize_filename(self, filename):
        sanitized_filename = re.sub(r"[\/\\\:\*\?\"\<\>\|\s]", "_", filename)
        sanitized_filename = "".join(c for c in sanitized_filename if c.isprintable())
        return sanitized_filename

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


class OhouseCrawler(BaseCrawler):
    def __init__(self, path, query, style=None, num_workers=8, verbose=False):
        super().__init__(path, "ohouse", num_workers, verbose)
        self.query = query
        self.style = style
        self.api_url = "https://ohou.se/cards/feed.json"

    def producer(self):
        page = 1
        while not self.finished:
            desks = self.fetch_desks(page)
            with self.lock:
                for desk in desks:
                    self.queue.put(desk)
            page += 1

    def fetch_desks(self, page):
        params = {
            "v": 5,
            "query": self.query,
            "search_affect_type": "Typing",
            "page": page,
            "per": 48,
        }
        if self.style:
            params["style"] = self.style

        response = requests.get(self.api_url, params=params, headers=self.headers)
        if response.status_code != 200:
            raise Exception("Failed to fetch data")

        desks = []
        response_data = response.json()
        fetched_desks = response_data["cards"]
        for fetched_desk in fetched_desks:
            desk = {
                "url": fetched_desk["image"]["url"],
                "name": str(fetched_desk["id"])
            }
            desks.append(desk)

        if not response_data["next"]:
            self.finished = True

        return desks
    

class PinterestCrawler(BaseCrawler):
    def __init__(self, path, query, num_workers=8, verbose=False):
        super().__init__(path, "pinterest", num_workers, verbose)
        self.query = query

        self.api_url = "https://pinterest.com/resource/BaseSearchResource/get/?"
        self.source_url = f"/search/pins/?q={query}"
        self.bookmark = ""

    def producer(self):
        while not self.finished:
            desks = self.fetch_desks()
            with self.lock:
                for desk in desks:
                    self.queue.put(desk)
        
    def fetch_desks(self):
        if self.bookmark == "":
            data = f'{{"options":{{"isPrefetch":false,"query":"{self.query}","scope":"pins","no_fetch_context_on_resource":false}},"context":{{}}}}'
        else:
            data = f'{{"options":{{"page_size":25,"query":"{self.query}","scope":"pins","bookmarks":["{self.bookmark}"],"field_set_key":"unauth_react","no_fetch_context_on_resource":false}},"context":{{}}}}'.strip()

        response = requests.get(self.api_url, params={"source_url": self.source_url, "data": data})
        if response.status_code != 200:
            raise Exception("Failed to fetch data")
        
        resource_response = response.json()["resource_response"]
        try:
            self.bookmark = resource_response["bookmark"]
        except KeyError:
            self.finished = True

        desks = []
        for fetched_desk in resource_response["data"]["results"]:
            desk = {
                "url": fetched_desk["images"]["orig"]["url"],
                "name": str(fetched_desk["id"])
            }
            desks.append(desk)
        return desks


if __name__ == "__main__":

    from config_loader import config

    train_path = config["PATH"]["train"]

    styles = ["모던", "북유럽", "빈티지", "내추럴", "프로방스&로맨틱", "클래식&앤틱", "한국&아시아", "유니크"]
    for i, style in enumerate(styles):
        path = os.path.join(train_path, style)
        desk_crawler = OhouseCrawler(path, "데스크테리어", style=i, verbose=True)
        desk_crawler.crawling()

    # desk_crawler = PinterestCrawler(train_path, "desk interior", verbose=True)
