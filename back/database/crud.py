from sqlalchemy.orm import Session
from database import models, schemas


def create_user(db: Session, user: schemas.UserBase):
    db_item = models.User(**user.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def check_user(db: Session, id:str, password=str):
    # 주어진 사용자 ID에 해당하는 정보가 데이터베이스에 있는지 확인
    existing_user = db.query(models.User).filter(models.User.id == id).first()
    
    if existing_user and existing_user.password != password:
        # 이미 존재하고 비밀번호도 일치하는 경우
        return "비밀번호가 틀렸습니다"
    elif existing_user and existing_user.password == password:
        return existing_user
    else:
        return "유저가 없음"