import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import re
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from sqlalchemy.orm import scoped_session
from database.database import SessionLocal, engine
from database.models import Base, DesignImages, ItemImages, DesignItemRelations


Base.metadata.create_all(bind=engine)


def create_retry_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


class OhouseDesignItemCrawler:
    def __init__(self, query):
        self.card_list_api_url = "https://ohou.se/cards/feed.json"
        self.card_api_url = "https://contents.ohou.se/api/card-collections/{}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.params = {
            "query": query,
            "search_affect_type": "Typing",
            "per": 20,
            "v": 5,
        }

        self.n_design = 0
        self.n_item = 0
        self.design_index = 0
        self.item_index = 0

        self.get_session = scoped_session(SessionLocal)

    def crawling(self):
        finished = False
        page = 1

        while not finished:
            self.params["page"] = page

            session = create_retry_session()
            response = session.get(
                self.card_list_api_url, params=self.params, headers=self.headers
            )
            page += 1
            if response.status_code != 200:
                continue

            response_data = response.json()
            for card in response_data["cards"]:
                try:
                    card_collection_id = card["card_collection"]["id"]
                    self.crawling_card(card_collection_id)

                except Exception as e:
                    continue

            if not response_data["next"]:
                finished = True

    def crawling_card(self, card_collection_id):
        session = create_retry_session()
        response = session.get(
            self.card_api_url.format(card_collection_id), headers=self.headers
        )

        response_data = response.json()
        if response_data["status"] != 200:
            return
    
        card = response_data["data"]["cards"][0]
        design = {
            "id": card_collection_id,
            "src_url": self.to_webp_url(card["image"]["url"]),
            "landing": f"https://contents.ohou.se/contents/card_collections/{card_collection_id}"
        }
        design_object = DesignImages(**design, index=self.design_index)
        ret = self.to_db(design_object)
        if ret:
            self.design_index += 1
        else:
            return

        for tag in card["tags"]:
            if tag["type"] != "product":
                continue

            product = tag["product"]
            item = {
                "id": product["id"],
                "name": product["name"],
                "src_url": self.to_webp_url(product["imageUrl"]),
                "landing": f"https://ohou.se/productions/{product['id']}/selling",
            }
            item_object = ItemImages(**item, index=self.item_index)
            ret = self.to_db(item_object)
            if ret:
                self.item_index += 1

            relation = DesignItemRelations(design_id=card_collection_id, item_id=item["id"])
            self.to_db(relation)

    def to_webp_url(self, url):
        if "amazon" in url:
            url = re.sub(r"\.s.*?\.com", "", url)
            url = url.replace("https://", "https://image.ohou.se/i/")
        url += "?gif=1&webp=1"
        return url

    def to_db(self, object):
        try:
            session = self.get_session()
            session.add(object)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            return False


if __name__ == "__main__":
    crawler = OhouseDesignItemCrawler("데스크테리어")
    crawler.crawling()