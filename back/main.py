from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List

import os
import shutil
import uuid

from detect import detect

from sqlalchemy.orm import Session
from database import crud
from database.models import *
from database.schemas import *
from database.database import SessionLocal, engine


Base.metadata.create_all(bind=engine)
app = FastAPI()
app.mount("/uploaded_images",
          StaticFiles(directory="uploaded_images"), name="uploaded_images")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 메인
@app.get("/")
async def root():
    return "hello world"


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용하려면 "*" 사용
    allow_headers=["*"],  # 모든 헤더를 허용하려면 "*" 사용
)
result_folder = "/Users/park_sh/Desktop/what-desk/back/result"
upload_folder = "/Users/park_sh/Desktop/what-desk/back/uploads"


@app.get("/get_image/{image_filename}")
async def get_image(image_filename: str):
    image_path = os.path.join(result_folder, image_filename)
    return FileResponse(image_path, media_type="image/jpeg")


# 이미지 업로드 및 처리 결과 반환
@app.post("/process_image/")
async def process_image(file: UploadFile):
    try:
        # 파일명을 UUID로 생성하여 유니크한 이름으로 저장
        filename = f"{str(uuid.uuid4())}.jpg"
        file_path = os.path.join(upload_folder, filename)

        # 업로드된 파일 저장
        with open(file_path, "wb") as image:
            image.write(file.file.read())
        # detect 함수를 적용하여 결과 반환
        result_image_path = detect(file_path, result_folder)

        # 결과물 파일명 반환
        result_filename = os.path.basename(result_image_path)
        return {"result_filename": result_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 유저정보
@app.post("/user/sign_up", response_model=User)
async def create_user(user: User, db: Session = Depends(get_db)):
    return await crud.create_user(db, user)


@app.post("/user/sign_in")
async def check_user(user: User, db: Session = Depends(get_db)):
    db_user = await crud.read_user_by_id(db, user.username)

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.password != db_user.password:
        raise HTTPException(status_code=400, detail="Incorrect password")

    return {"message": "User exist"}


# posting test
@app.post("/post/create", response_model=Post)
async def create_post(post: PostForm, db: Session = Depends(get_db)):
    return await crud.create_post(db, post)


@app.post("/comment/create", response_model=Comment)
async def create_comment(comment: Comment, db: Session = Depends(get_db)):
    return await crud.create_comment(db, comment)


@app.get("/post/{post_id}", response_model=Post)
async def read_post(post_id: int, db: Session = Depends(get_db)):
    post = crud.increment_view_count(db. post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post does not exist.")


@app.get("/post/search", response_model=List[Post])
async def search_post(
    category: str | None = None,
    author_id: str | None = None,
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    posts = await crud.search_posts(
        db, author_id=author_id, category=category, keyword=keyword
    )
    return posts


@app.get("/comment/search", response_model=List[Comment])
async def search_comment(
    post_id: str | None = None,
    author_id: str | None = None,
    db: Session = Depends(get_db),
):
    comments = await crud.read_comment(
        db, author_id=author_id, post_id=post_id
    )
    return comments


@app.post("/like", response_model=Post)
async def like_post(author_id: str, post_id: str, db: Session = Depends(get_db)):
    await crud.create_like(db, author_id, post_id)
    return await crud.increment_like_count(db, post_id)


# image test
@app.post("/image/upload", response_model=Image)
async def upload_image(file: UploadFile, db: Session = Depends(get_db)):
    os.makedirs("uploaded_images", exist_ok=True)

    image_id = str(uuid.uuid4())
    filename = file.filename
    file_extension = os.path.splitext(filename)[1]

    with open(f"uploaded_images/{image_id}{file_extension}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image(image_id=image_id, filename=filename)
    return await crud.created_image(db, image)


# test
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000
# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????
