import os
# from back.recommend_system import recommend_by_uservector
from fastapi import FastAPI, UploadFile, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import recommend, user, chat, community, images

from database.database import Base, engine

import faiss

from detect import detect
from process_image import process
from img2vec import Feat2Vec, Obj2Vec
from config_loader import config

import plotly.express as px


Base.metadata.create_all(bind=engine)
app = FastAPI()
app.include_router(recommend.router)
app.include_router(user.router)
app.include_router(chat.router)
app.include_router(community.router)
app.include_router(images.router)

os.makedirs(config["PATH"]["upload"], exist_ok=True)
os.makedirs(config["PATH"]["result"], exist_ok=True)
os.makedirs(config["PATH"]["train"], exist_ok=True)

# app.mount(
#     "/images/upload",
#     StaticFiles(directory=config["PATH"]["upload"]),
#     name="uploaded_images",
# )
# app.mount(
#     "/images/result",
#     StaticFiles(directory=config["PATH"]["result"]),
#     name="result_images",
# )
# app.mount(
#     "/images/train",
#     StaticFiles(directory=config["PATH"]["train"]),
#     name="train_images",
# )
# app.mount(
#     "/images/default",
#     StaticFiles(directory=config["PATH"]["default"]),
#     name="default_images",
# )


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


# 이미지 업로드 및 처리 결과 반환
@app.post("/process_image")
async def process_image(file: UploadFile):
    return process(file)


@app.post("/prototype_process")
async def prototype_process(file: UploadFile):
    file_path = os.path.join(config["PATH"]["upload"], file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # how to use Img2Vec module to get nearest image
    feat2vec = Feat2Vec()
    feat_vec = feat2vec.get_vector(file_path)
    feat_idx = faiss.read_index("vectors/vgg_features.index")
    _, feat_result = feat_idx.search(feat_vec, 5)

    # example
    # obj2vec = Obj2Vec()
    # obj_vec = obj2vec.get_vector(file_path)
    # obj_idx = faiss.read_index("vectors/object_counts.index")
    # _, obj_result = obj_idx.search(obj_vec, 5)

    result = []
    image_dir = config["PATH"]["train"]
    image_paths = os.listdir(image_dir)

    for i in feat_result[0]:
        result.append("/" + os.path.join(image_dir, image_paths[i]))

    df = px.data.tips()
    fig = px.box(df, x="day", y="total_bill", color="smoker")
    fig.update_traces(quartilemethod="inclusive")
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    return {"file_name": result, "plot": plot_html}





# webhook check
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000
# test
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????
