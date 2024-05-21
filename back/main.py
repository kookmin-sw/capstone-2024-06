import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import recommend, user, chat, community, images

from database.database import Base, engine
from config_loader import config


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


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용하려면 "*" 사용
    allow_headers=["*"],  # 모든 헤더를 허용하려면 "*" 사용
)


# webhook check
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000
# test
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????
