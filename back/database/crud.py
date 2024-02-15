from sqlalchemy.orm import Session
from database import models, schemas


async def create_user(db: Session, user: schemas.UserSignUp):
    db_item = models.User(**user.model_dump())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

async def read_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()