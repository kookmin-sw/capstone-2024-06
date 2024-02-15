from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import uuid

from detect import detect

from sqlalchemy.orm import Session
from database import crud, models, schemas
from database.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

SECRET_KEY = "5620f1628ee220e31ae6d0a18dee3d3e268def2b8847df798a419a6d93adedb3" # temp
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="user/oauth_sign_in")


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


async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await crud.read_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

          
# 메인
@app.get("/")
async def root():
    return "hello world"

origins = [
    "*"
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
@app.post("/user/sign_up", response_model=schemas.User)
def create_user(user: schemas.User, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)

@app.post("/user/sign_in")
def check_user(user: schemas.User, db: Session = Depends(get_db)):
    db_user = crud.read_user_by_id(db=db, id=user.id)

    if not db_user:
        raise HTTPException(status_code=404, detail='User not found')
    
    if user.password != db_user.password:
        raise HTTPException(status_code=400, detail='Incorrect password')
    
    return {"message": "User exist"}


# OAuth2 test
@app.post("/user/oauth_sign_up", response_model=schemas.User)
async def oauth_create_user(user: schemas.UserSignUp, db: Session = Depends(get_db)):
    hashed_password = get_hashed_password(user.password)
    user = schemas.UserDB(**user.model_dump(), hashed_password=hashed_password)
    return await crud.create_user(db=db, user=user)

@app.post("/user/oauth_sign_in")
async def oauth_sign_in(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> schemas.Token:
    user = await authenticate_user(db, form_data.username, form_data.password)
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
async def me(current_user: schemas.User = Depends(get_current_user)):
    return current_user

# test
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000 
# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????