import os
import shutil
import uuid
import base64
import json
import requests
import re
from typing import List
from fastapi import FastAPI, Request
from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext
from back.recommend_system import recommend_by_uservector
from fastapi import (
    FastAPI,
    UploadFile,
    HTTPException,
    Depends,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from database import crud
from database.models import *
from database.schemas import *
from database.database import SessionLocal, engine

import faiss

from detect import detect
from process_image import process
from img2vec import Feat2Vec, Obj2Vec
from config_loader import config

import plotly.express as px


Base.metadata.create_all(bind=engine)

SECRET_KEY = "secret"  # temp
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


Base.metadata.create_all(bind=engine)
app = FastAPI()

os.makedirs(config["PATH"]["upload"], exist_ok=True)
os.makedirs(config["PATH"]["result"], exist_ok=True)
os.makedirs(config["PATH"]["train"], exist_ok=True)

app.mount(
    "/images/upload",
    StaticFiles(directory=config["PATH"]["upload"]),
    name="uploaded_images",
)
app.mount(
    "/images/result",
    StaticFiles(directory=config["PATH"]["result"]),
    name="result_images",
)
app.mount(
    "/images/train",
    StaticFiles(directory=config["PATH"]["train"]),
    name="train_images",
)
app.mount(
    "/images/default",
    StaticFiles(directory=config["PATH"]["default"]),
    name="default_images",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(password: str, hashed_password: str):
    return pwd_context.verify(password, hashed_password)


def get_hashed_password(password: str):
    return pwd_context.hash(password)


async def authenticate_user(db: Session, user: UserSignIn):
    user_db = await crud.read_user_by_id(db, user.user_id)
    if not user_db:
        return False
    if not verify_password(user.password, user_db.hashed_password):
        return False
    return user_db


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_jwt_payload(token):
    header, payload, signature = token.split(".")

    decoded_payload = base64.urlsafe_b64decode(payload + "==").decode("utf-8")
    parsed_payload = json.loads(decoded_payload)

    return parsed_payload


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_jwt_payload(token)
        jwt.decode(token, SECRET_KEY, ALGORITHM)
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_if_signed_in(token: str | None = Depends(optional_oauth2_scheme)):
    try:
        if not token or token == "undefined":
            return None

        payload = decode_jwt_payload(token)
        jwt.decode(token, SECRET_KEY, ALGORITHM)
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


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

# 사용자 선택 기반 이미지 추천
@app.post("/recommend_image")
async def recommend_image(request: Request):
    data = await request.json()
    user_id = data["user_id"]
    selected_images = data["selected_images"]
    return recommend_by_uservector(user_id, selected_images)

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


@app.post("/user")
async def create_user(user: UserForm, db: Session = Depends(get_db)):
    hashed_password = get_hashed_password(user.password)
    user = HashedUser(**user.model_dump(), hashed_password=hashed_password)

    user_db = await crud.read_user_by_id(db, user.user_id)
    if user_db:
        raise HTTPException(status_code=409, detail="User ID already exists")

    await crud.create_user(db, user)
    return {"message": "User created successfully"}


@app.post("/token", response_model=TokenResult)
async def generate_token(user: UserSignIn, db: Session = Depends(get_db)):
    user = await authenticate_user(db, user)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    data = {"sub": user.user_id}
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=data, expires_delta=access_token_expires)
    return {"user": user, "access_token": access_token}


@app.post("/token/{access_token}", response_model=TokenResult)
async def generate_token_from_external_provider(
    access_token: str, provider: str, user: UserInfo, db: Session = Depends(get_db)
):
    if provider == "google":
        response = requests.get(
            "https://www.googleapis.com/oauth2/v1/tokeninfo",
            params={"access_token": access_token},
        )
    elif provider == "kakao":
        response = requests.get(
            "https://kapi.kakao.com/v1/user/access_token_info",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    elif provider == "naver":
        response = requests.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    else:
        raise HTTPException(status_code=501, detail="Provider not supported")

    if not response.ok:
        raise HTTPException(status_code=401, detail="Invalid access token")

    user_external_map = await crud.read_user_external_map(db, user.user_id, provider)

    if user_external_map:
        user = user_external_map.user
    else:
        internal_id = str(uuid.uuid4())
        user_external_map = UserExternalMap(
            external_id=user.user_id, provider=provider, user_id=internal_id
        )
        user.user_id = internal_id

        try:
            await crud.create_user(db, user)
        except IntegrityError as e:
            error_message = e.orig.diag.message_detail
            pattern = r"\((.*?)\)=\((.*?)\)"
            matched = re.search(pattern, error_message)
            key = matched.group(1)

            if key == "email":
                raise HTTPException(status_code=409, detail="Email already registerd")
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail="Internal server error")

        await crud.create_user_external_map(db, user_external_map)

    data = {"sub": user.user_id}
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=data, expires_delta=access_token_expires)
    return {"user": user, "access_token": access_token}


@app.get("/user/me", response_model=UserInfo)
async def current_user(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    return await crud.read_user_by_id(db, user_id)


@app.get("/user/{user_id}", response_model=UserInfo)
async def get_user_profile(user_id: str, signed_in_user_id: str = Depends(get_current_user_if_signed_in), db: Session = Depends(get_db)):
    user = await crud.read_user_by_id(db, user_id, signed_in_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@app.get("/post/temp", response_model=TempPost)
async def create_temporary_code(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    temp_post = await crud.create_temp_post(db, user_id)
    return temp_post


@app.post("/post/{temp_post_id}")
async def create_post(
    temp_post_id: int,
    post: PostForm,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    await crud.create_post(db, post, user_id, temp_post_id)
    return {"message": "Post created successfully"}


@app.get("/post/search", response_model=list[PostPreview])
async def search_posts(
    category: str | None = None,
    author_id: str | None = None,
    keyword: str | None = None,
    order: str = "newest",
    per: int = 24,
    page: int = 1,
    user_id: str | None = Depends(get_current_user_if_signed_in),
    db: Session = Depends(get_db),
):
    if order not in ["newest", "most_viewed", "most_scrapped", "most_liked"]:
        return HTTPException(status_code=400, detail="Invalid order parameter")

    posts = await crud.search_posts(
        db,
        category=category,
        author_id=author_id,
        keyword=keyword,
        order=order,
        per=per,
        page=page,
        user_id=user_id,
    )
    return posts


@app.get("/post/{post_id}", response_model=Post)
async def read_post(
    post_id: int,
    user_id: str | None = Depends(get_current_user_if_signed_in),
    db: Session = Depends(get_db),
):
    post = await crud.read_post_with_view(db, post_id, user_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post does not exist.")
    return post


@app.get("/comment/{post_id}", response_model=list[Comment])
async def read_comment(post_id: int, db: Session = Depends(get_db)):
    comments = await crud.read_comments(db, post_id)
    return comments


@app.delete("/post/{post_id}")
async def delete_post(
    post_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if post.author_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Permission denied: You are not the author of this post",
        )

    await crud.delete_post(db, post)
    return {"message": "Post deleted successfully"}


@app.post("/scrap/post/{post_id}")
async def scrap_post(
    post_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    await crud.create_post_scrap(db, user_id, post_id)
    return {"message": "User scrapped post successfully"}


@app.post("/like/post/{post_id}")
async def scrap_post(
    post_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    await crud.create_post_like(db, user_id, post_id)
    return {"message": "User liked post successfully"}


@app.post("/scrap/comment/{comment_id}")
async def scrap_comment(
    comment_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    comment = await crud.read_comment(db, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    await crud.create_comment_scrap(db, user_id, comment_id)
    return {"message": "User scrapped comment successfully"}


@app.post("/comment")
async def create_comment(
    comment: CommentForm,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, comment.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    await crud.create_comment(db, comment, user_id)
    return {"messaeg": "Comment created successfully"}


@app.delete("/comment/{comment_id}")
async def delete_comment(
    comment_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    comment = await crud.read_comment(db, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    if not comment.author_id:
        raise HTTPException(status_code=404, detail="Comment already deleted")

    if comment.author_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="You are not the author of this comment",
        )

    await crud.delete_comment(db, comment)
    return {"message": "Comment deleted successfully"}


@app.post("/image/{temp_post_id}", response_model=Image)
async def upload_image(
    temp_post_id: int,
    file: UploadFile,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    upload_path = config["PATH"]["upload"]
    filename = file.filename
    file_extension = os.path.splitext(filename)[1]
    file_path = os.path.join(upload_path, str(uuid.uuid4()) + file_extension)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_id = "/" + file_path
    image = Image(image_id=image_id, filename=filename)
    return await crud.create_image(db, image, temp_post_id)


@app.post("/follow/{followee_user_id}")
async def follow_user(
    followee_user_id: str,
    follower_user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    await crud.create_follow(db, follower_user_id, followee_user_id)
    return {"message": "Followed successfully"}


@app.delete("/follow/{followee_user_id}")
async def unfollow_user(
    followee_user_id: str,
    follower_user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    follow = await crud.read_follow(db, follower_user_id, followee_user_id)
    if not follow:
        raise HTTPException(status_code=404, detail="You are not following this user")

    await crud.delete_follow(db, follow)
    return {"message": "Unfollowed successfully"}


@app.post("/followers", response_model=list[UserInfo])
async def get_followers(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    user = await crud.read_user_by_id(db, user_id)
    return user.followers


@app.post("/followees", response_model=list[UserInfo])
async def get_followees(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    user = await crud.read_user_by_id(db, user_id)
    return user.followees


@app.post("/scrapped_posts", response_model=list[PostPreview])
async def get_scrapped_posts(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    posts = await crud.search_posts(db, user_id=user_id, scrapped=True)
    return posts


@app.put("/user/modification", response_model=UserInfo)
async def modify_user_profile(
    user_profile: UserProfile,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = await crud.modify_user(db, user_id, user_profile)
    return user


@app.put("/user/modification/profile_image", response_model=UserInfo)
async def modify_user_profile_image(
    file: UploadFile,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    upload_path = config["PATH"]["upload"]
    filename = file.filename
    file_extension = os.path.splitext(filename)[1]
    file_path = os.path.join(upload_path, str(uuid.uuid4()) + file_extension)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    image_id = config["URL"]["api_url"] + "/" + file_path
    user_profile = UserProfile(image=image_id)
    user = await crud.modify_user(db, user_id, user_profile)
    return user


@app.get("/notification", response_model=list[Notification])
async def get_notifications(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    notifications = await crud.read_notifications(db, user_id)
    return notifications


@app.post("/notification/{notification_id}", response_model=Notification)
async def check_notification(
    notification_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    notification = await crud.check_notification(db, notification_id)
    return notification


@app.delete("/notification")
async def delete_notification(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    await crud.delete_notifications(db, user_id)
    return {"message": "Notification deleted successfully"}


connections = dict()


@app.websocket("/chat/{opponent_id}")
async def chatting_websocket(
    opponent_id: str, websocket: WebSocket, db: Session = Depends(get_db)
):
    await websocket.accept()
    token = await websocket.receive_text()
    user_id = get_current_user(token)

    if user_id in connections:
        del connections[user_id]
    connections[user_id] = websocket

    try:
        while True:
            message = await websocket.receive_text()
            chat_history = BaseChatHistory(
                message=message, sender_id=user_id, receiver_id=opponent_id
            )
            await crud.create_chat_history(db, chat_history)
            if opponent_id in connections:
                print(f"{chat_history.message} to {opponent_id}")
                await connections[opponent_id].send_text(chat_history.model_dump_json())

    except WebSocketDisconnect:
        ...
    finally:
        if user_id in connections:
            del connections[user_id]
        connections[user_id] = websocket


@app.get("/chat/{opponent_id}", response_model=list[ChatHistory])
async def get_chat_histories(
    opponent_id: str,
    last_chat_history_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat_histories = await crud.read_chat_histories(
        db, user_id, opponent_id, last_chat_history_id
    )
    return chat_histories


@app.get("/chat_rooms", response_model=list[ChatRoom])
async def get_chatting_rooms(user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    opponents = await crud.read_chatting_rooms(db, user_id)
    return opponents


# webhook check
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000
# test
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????
