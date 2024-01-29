from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import uuid

app = FastAPI()

result_folder = "/Users/park_sh/Desktop/what-desk/back/result"
upload_folder = "/Users/park_sh/Desktop/what-desk/uploads"

@app.get("/get_image/{image_filename}")
async def get_image(image_filename: str):
    image_path = os.path.join(result_folder, image_filename)
    return FileResponse(image_path, media_type="image/jpeg")

# 이미지 업로드 및 결과 반환 라우트
@app.post("/process_image/")
async def process_image(file: UploadFile):
    try:
        # 파일명을 UUID로 생성하여 유니크한 이름으로 저장
        filename = f"{str(uuid.uuid4())}.jpg"
        file_path = os.path.join(upload_folder, filename)

        # 업로드된 파일 저장
        with open(file_path, "wb") as image:
            image.write(file.file.read())

        return {"filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
