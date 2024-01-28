from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

# 템플릿을 사용하기 위한 설정
# templates = Jinja2Templates(directory="templates")

# 이미지 파일이 저장된 폴더
result_folder = "/Users/park_sh/Desktop/what-desk/back/result"
# os.makedirs(image_folder, exist_ok=True)

# 이미지 파일을 반환하는 라우트
@app.get("/get_image/{image_filename}")
async def get_image(image_filename: str):
    image_path = os.path.join(result_folder, image_filename)
    return FileResponse(image_path, media_type="image/jpeg")

# # 이미지 업로드 및 결과 반환 라우트
# @app.post("/process_image/")
# async def process_image(file: UploadFile = File(...)):
#     try:
#         # 업로드된 이미지를 저장
#         file_path = os.path.join(image_folder, file.filename)
#         with open(file_path, "wb") as image:
#             shutil.copyfileobj(file.file, image)

#         return {"message": "Image uploaded successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/add")
# async def add_numbers(num1: int, num2: int):
#     total = sum(num1,num2)
#     return {"result": total}

# def sum(a,b):
#     return a+b