import os
import uuid
from detect import detect
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from recommended_image import recommend_image

upload_folder = "./uploaded_images"

# 파일명을 UUID로 생성하여 유니크한 이름으로 저장
def process(file = UploadFile):

    filename = f"{str(uuid.uuid4())}.jpg"

    file_path = os.path.join(upload_folder, filename)

    # 업로드된 파일 저장
    with open(file_path, "wb") as image:
        image.write(file.file.read())
    # 함수를 적용하여 결과 반환
    result_image_path = recommend_image(file_path)

    # 결과물 파일명 반환
    return {"file_name ": result_image_path}
