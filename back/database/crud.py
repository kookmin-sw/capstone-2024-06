from sqlalchemy.orm import Session
from database import models, schemas


def create_user(db: Session, user: schemas.User):
    user = models.User(**user.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_post(db: Session, post: schemas.Post):
    post = models.Post(**post.model_dump())
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def create_comment(db: Session, comment: schemas.Comment):
    comment = models.Comment(**comment.model_dump())
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


def read_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def read_post(
    db: Session, category: str = None, writer_username: str = None, keyword: str = None
):
    query = db.query(models.Post)

    if category:
        query = query.filter(models.Post.category == category)

    if writer_username:
        query = query.filter(models.Post.writer_username == writer_username)

    if keyword:
        query = query.filter(models.Post.title.ilike(f"%{keyword}%"))

    return query.all()


def read_comment(db: Session, post_id: str = None, writer_username: str = None):
    query = db.query(models.Comment)

    if post_id:
        query = query.filter(models.Comment.post_id == post_id)

    if writer_username:
        query = query.filter(models.Comment.writer_username == writer_username)

    return query.all()
