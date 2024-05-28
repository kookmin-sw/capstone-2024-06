import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from pillow_heif import register_heif_opener

register_heif_opener()

from database.models import DesignImages
from dependencies import get_db


def download_image(design):
    index, src_url = design
    try:
        response = requests.get(src_url, stream=True)
        response.raise_for_status()
        filename = f"dataset/{index}.jpg"

        image = Image.open(BytesIO(response.content))
        image = image.resize((224, 224))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(filename)

        with Image.open(filename) as image:
            assert image.size == (224, 224)
            assert image.mode == "RGB"
            image.verify()

    except requests.RequestException as e:
        print(f"Failed to download {filename} from {src_url}: {e}")
    except IOError as e:
        print(f"Failed to process image {filename} from {src_url}: {e}")
    except AssertionError as e:
        print(f"Failed to verify image {filename} from {src_url}: {e}")
    except Exception as e:
        print(f"Failed on image {filename} from {src_url}: {e}")


session = next(get_db())
designs = session.query(DesignImages).order_by(DesignImages.index).all()
designs = [(design.index, design.src_url) for design in designs]

designs = [designs[i] for i in [776, 1404, 3334, 4157, 4639]]

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(download_image, designs), total=len(designs)))
