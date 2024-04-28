import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from database.models import *
from database.database import SessionLocal


def insert_index(session, i, image_path):
    image_name = os.path.basename(image_path)
    image = session.query(DesignImages).filter(DesignImages.filename == image_name).first()
    image.index = i
    session.commit()


if __name__ ==  "__main__":
    json_path = sys.argv[1]

    session = SessionLocal()
    with open(json_path, "r") as f:
        image_paths = json.load(f)
    
    for i, image_path in enumerate(image_paths):
        insert_index(session, i, image_path)