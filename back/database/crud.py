from sqlalchemy.orm import Session
from database import models, schemas


def create_user(db: Session, user: schemas.UserBase):
    db_item = models.User(**user.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def check_user(db: Session, id:str, password=str):
    # id 확인
    existing_user = db.query(models.User).filter(models.User.id == id).first()
    
    if existing_user and existing_user.password != password:
        return "비밀번호가 틀렸습니다"
    elif existing_user and existing_user.password == password:
        return existing_user
    else:
        return "유저가 없음"