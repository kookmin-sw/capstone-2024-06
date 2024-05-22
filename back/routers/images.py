import os
import io
from PIL import Image

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from dependencies import *


router = APIRouter(
    prefix="/images",
    tags=["images"]
)


@router.get("/{image_path:path}")
async def get_image(image_path: str, w: int | None = None, h: int | None = None):
    image_path = os.path.join("images", image_path)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Invalid image path")
    
    img = Image.open(image_path)
    img_format = img.format
    if w and h:
        img = img.resize((w, h))

    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img_format)
    img_byte_arr.seek(0)

    headers = {
        "Cache-Control": "public, max-age=86400"
    }

    return StreamingResponse(img_byte_arr, media_type=f"image/{img_format.lower()}", headers=headers)
