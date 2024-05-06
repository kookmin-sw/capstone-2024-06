import os
import faiss
import numpy as np
from fastapi import APIRouter, UploadFile, Depends

from sqlalchemy.orm import Session

from database import crud
from database.models import *
from database.schemas import *
from img2vec import Feat2Vec
from models.VGGAutoEncoder import LightVGGAutoEncoder

from dependencies import *


router = APIRouter(
    prefix="/recommend",
    tags=["recommend"]
)


@router.get("/sample", response_model=list[DesignImage])
async def get_images_to_rate(n: int = 5, db: Session = Depends(get_db)):
    return await crud.read_random_design_images(db, n)


@router.post("/preference", response_model=list[DesignImage])
async def recommend_by_preference(rated_images: list[RatedImage], user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    feat_idx = faiss.read_index("vectors/features.index")

    weighted_vectors = []
    weight_mapping = [-0.2, 0.2, 1.0, 1.2, 2.0]

    weights = 0
    for rated_image in rated_images:
        vector = feat_idx.reconstruct(rated_image.index)
        weight = weight_mapping[rated_image.rating-1]
        weighted_vectors.append(weight * vector)
        weights += weight
    weighted_vector = np.mean(weighted_vectors, axis=0)

    user = await crud.read_user_by_id(db, user_id)
    user_vector = user.embedding
    user_vector = None

    user_vector = weighted_vector if user_vector is None else (user_vector + weighted_vector) * 0.5
    user.embedding = user_vector
    db.commit()

    design_images = []
    _, feat_result = feat_idx.search(np.expand_dims(user_vector, axis=0), 5)
    for i in feat_result[0]:
        design_image = await crud.read_design_images(db, int(i))
        design_images.append(design_image)

    return design_images


@router.post("/image", response_model=list[DesignImage])
async def recommend_by_source_image(file: UploadFile, user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    file_path = os.path.join(config["PATH"]["upload"], file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    autoencoder = LightVGGAutoEncoder.load_from_checkpoint(config["PATH"]["model"])
    encoder = autoencoder.model.encoder
    feat2vec = Feat2Vec(encoder)
    feat_vec = feat2vec.get_vector(file_path)

    feat_idx = faiss.read_index("vectors/features.index")
    design_images = []
    _, feat_result = feat_idx.search(np.expand_dims(feat_vec, axis=0), 5)
    for i in feat_result[0]:
        design_image = await crud.read_design_images(db, int(i))
        design_images.append(design_image)
    
    return design_images
