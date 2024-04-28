import faiss
import numpy as np
from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from database import crud
from database.models import *
from database.schemas import *

from dependencies import *


recommend_router = APIRouter(
    prefix="/recommend",
    tags=["recommend"]
)


@recommend_router.get("/sample_images", response_model=list[DesignImage])
async def get_images_to_rate(n: int = 5, db: Session = Depends(get_db)):
    return await crud.read_random_design_images(db, n)


@recommend_router.post("/", response_model=list[DesignImage])
async def recommend_image(rated_images: list[RatedImage], user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    feat_idx = faiss.read_index("vectors/features.index")

    weighted_vectors = []
    weight_mapping = [-0.2, 0.2, 1.0, 1.2, 2.0]

    for rated_image in rated_images:
        vector = feat_idx.reconstruct(rated_image.index)
        weight = weight_mapping[rated_image.rating-1]
        weighted_vectors.append(weight * vector)
    weighted_vector = np.mean(weighted_vectors, axis=0)

    user = await crud.read_user_by_id(db, user_id)
    user_vector = user.embedding

    user_vector = weighted_vector if user_vector is None else (user_vector + weighted_vector) * 0.5
    user.embedding = user_vector
    db.commit()

    design_images = []
    _, feat_result = feat_idx.search(np.expand_dims(user_vector, axis=0), 5)
    for i in feat_result[0]:
        design_image = await crud.read_design_images(db, int(i))
        design_images.append(design_image)

    return design_images