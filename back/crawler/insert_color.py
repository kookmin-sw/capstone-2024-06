import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from tqdm import tqdm
from pillow_heif import register_heif_opener

register_heif_opener()

from database.models import ItemImages
from dependencies import get_db


def download_image(item):
    index, src_url = item
    try:
        response = requests.get(src_url, stream=True)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        color = extract_color(image)
        colors[index] = color

    except requests.RequestException as e:
        print(f"Failed to download {index} from {src_url}: {e}")
    except IOError as e:
        print(f"Failed to process image {index} from {src_url}: {e}")
    except AssertionError as e:
        print(f"Failed to verify image {index} from {src_url}: {e}")
    except Exception as e:
        print(f"Failed on image {index} from {src_url}: {e}")

def extract_color(image):
    image = np.array(image)
    image = cv2.resize(image, (100, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 100)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max(1, len(contours)//2)]
    if len(contours) == 0:
        center_color = image[image.shape[0] // 2, image.shape[1] // 2]
        return center_color

    mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_without_bg = cv2.bitwise_and(image, image, mask=mask)

    colors = image_without_bg[image_without_bg.sum(axis=2) != 0]
    main_color = np.median(colors, axis=0).astype(np.uint8)
    return main_color


session = next(get_db())
items = session.query(ItemImages).order_by(ItemImages.index).all()
item_mapped = [(item.index, item.src_url) for item in items]
colors = [None] * len(items)

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(download_image, item_mapped), total=len(items)))

for item in items:
    item.color = colors[item.index]
session.commit()
