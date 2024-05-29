import os
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, Depends

import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from sqlalchemy.orm import Session
from sqlalchemy import func

from database import crud
from database.models import *
from database.schemas import *
from img2vec import Feat2Vec

import torchvision.models as models
from models.bpr_model import load_model, recommend_items

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
    await crud.create_or_update_ratings(db, user_id, rated_images)

    feature_mat = np.load("vectors/features.npy")
    rated_images_index = [rated_image.index for rated_image in rated_images]
    query_mat = feature_mat[np.array(rated_images_index)]
    preference_mat = np.array([[rated_image.rating for rated_image in rated_images]])

    similarity_mat = query_mat @ feature_mat.T
    result_mat = preference_mat @ similarity_mat
    result_index = result_mat[0].argsort()[::-1]

    design_images = []
    i = 0
    while len(design_images) < 5:
        index = result_index[i]
        if index not in rated_images_index:
            design_image = await crud.read_design_images(db, int(index))
            design_images.append(design_image)
        i += 1

    await crud.update_analysis_history(db, user_id, [DesignImage.model_validate(design_image).model_dump_json() for design_image in design_images])
    
    return design_images


@router.post("/image", response_model=list[DesignImage])
async def recommend_by_source_image(file: UploadFile, user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    file_path = os.path.join(config["PATH"]["upload"], file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier = model.classifier[:-1]
    feat2vec = Feat2Vec(model, resize=(224, 224))
    query_vec = feat2vec.get_vector(file_path)

    feature_mat = np.load("vectors/features.npy")
    query_mat = query_vec.reshape(1, -1)
    similarity_mat = query_mat @ feature_mat.T

    design_images = []
    for i in similarity_mat[0].argsort()[-1:-6:-1]:
        design_image = await crud.read_design_images(db, int(i))
        design_images.append(design_image)

    await crud.update_analysis_history(db, user_id, [DesignImage.model_validate(design_image).model_dump_json() for design_image in design_images])

    return design_images


@router.get("/item")
async def recoomend_items_by_color(index: int, user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    image = await crud.read_design_images(db, index)
    url = image.src_url

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    response = session.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data")
    
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    colors = np.uint8(centers)
    color_result = []
    for color in colors:
        color = color.tolist()
        items = await crud.read_item_images(db, color)
        color_result.append({
            "color": '#{:02x}{:02x}{:02x}'.format(*color),
            "items": [ItemImage.model_validate(item) for item in items]
        })
        
    n_designs = db.query(func.max(DesignImages.index)).scalar() + 1
    n_items = db.query(func.max(ItemImages.index)).scalar() + 1
    feature_mat = np.load("vectors/features.npy")
    model = load_model(n_designs, n_items, feature_mat)

    recommend_index = recommend_items(model, index, n_items)
    recommend_result = []
    for idx in recommend_index:
        item_image = await crud.read_item_images_by_idx(db, idx)
        recommend_result.append(ItemImage.model_validate(item_image))

    result = {
        "color": color_result,
        "recommend": recommend_result
    }
    return result


@router.post("/reload", response_model=list[DesignImage])
async def reload_analysis_history(user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    analysis_history = await crud.read_analysis_history(db, user_id)
    if analysis_history is None:
        raise HTTPException(status_code=400, detail="Analysis history does not exist")

    return [DesignImage.model_validate_json(design_image) for design_image in analysis_history.history]