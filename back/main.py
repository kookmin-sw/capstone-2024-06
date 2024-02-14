from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

from detect import detect
from sqlalchemy.orm import Session
from database import crud, models, schemas
from database.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


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
@app.post("/user/sign_up", response_model=schemas.UserBase)
def create_user(user: schemas.UserBase, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)

@app.post("/user/sign_in")
def check_user(user: schemas.UserBase, db: Session = Depends(get_db)):
    db_user = crud.read_user_by_id(db=db, id=user.id)

    if not db_user:
        raise HTTPException(status_code=404, detail='User not found')
    
    if user.password != db_user.password:
        raise HTTPException(status_code=400, detail='Incorrect password')
    
    return {"message": "User exist"}


# test2
# 서버 오픈 ->  uvicorn main:app --reload --host 0.0.0.0 --port 8000 
# 가상환경 -> source venv/bin/activate, 종료 -> deactivate
# db -> db 실행(brew services start postgresql), db 확인(psql -U admin -d mydb), db 종료(brew services stop postgresql), SELECT * FROM users;
# http://210.178.142.51:????
