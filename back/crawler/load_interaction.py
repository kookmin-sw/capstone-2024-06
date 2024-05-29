import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import csv
from database.models import DesignImages, ItemImages, DesignItemRelations
from dependencies import get_db


session = next(get_db())

interactions = (
    session.query(DesignImages.index, ItemImages.index)
    .join(DesignItemRelations, DesignItemRelations.design_id == DesignImages.id)
    .join(ItemImages, DesignItemRelations.item_id == ItemImages.id)
    .all()
)

with open("interactions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["design", "item"])

    for design_index, item_index in interactions:
        writer.writerow([design_index, item_index])