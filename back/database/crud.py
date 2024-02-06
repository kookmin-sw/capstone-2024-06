from sqlalchemy.orm import Session
from database import models, schemas


def create_user(db: Session, user: schemas.UserBase):
    db_item = models.User(**user.model_dump())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def read_user_by_id(db: Session, id: str):
    return db.query(models.User).filter(models.User.id == id).first()