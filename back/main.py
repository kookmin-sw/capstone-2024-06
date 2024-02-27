from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
import os
import uuid
import base64
import json

from detect import detect
from process_image import process
from sqlalchemy.orm import Session
from database import crud, models, schemas
from database.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

SECRET_KEY = "secret" # temp
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


app = FastAPI()

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()
          

def verify_password(password, hashed_password):
    return pwd_context.verify(password, hashed_password)


def get_hashed_password(password):
    return pwd_context.hash(password)


def decode_jwt_payload(token):
     header, payload, signature = token.split(".")
     decoded_payload = base64.urlsafe_b64decode(payload + "==").decode("utf-8")
     parsed_payload = json.loads(decoded_payload)
     return parsed_payload


async def authenticate_user(db, username: str, password: str):
    user = await crud.read_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

          
# 메인
@app.get("/")
async def root():
    return "hello world"

origins = [
    "http://localhost:3000"
]

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
    return process(file)
    

@app.post("/user/sign_up", response_model=schemas.User)
async def create_user(user: schemas.UserSignUp, db: Session = Depends(get_db)):
    hashed_password = get_hashed_password(user.password)
    user = schemas.UserDB(**user.model_dump(), hashed_password=hashed_password)
    return await crud.create_user(db=db, user=user)


@app.post("/token")
async def token(user: schemas.UserSignUp, db: Session = Depends(get_db)):
    user = await authenticate_user(db, user.username, user.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return schemas.Token(access_token=access_token, token_type="bearer")


@app.get("/user/me")
async def me(token: str = Depends(oauth2_scheme)):
    print(f"received token : {token}")

    payload = decode_jwt_payload(token)
    print(f"decoded payload : {payload}")

    try:
        validation = jwt.decode(token, SECRET_KEY, ALGORITHM)
    except JWTError:
         raise HTTPException(status_code=401, detail="Invalid token")

    return payload

# test asd
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000 
# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????
