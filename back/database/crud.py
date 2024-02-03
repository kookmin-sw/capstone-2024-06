from sqlalchemy.orm import Session
from database import models, schemas


def create_user(db: Session, user: schemas.UserBase):
    db_item = models.User(**user.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item