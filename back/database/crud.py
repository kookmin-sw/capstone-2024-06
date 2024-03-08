from sqlalchemy import and_, or_
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.sql import alias, select, column
from database.models import *
from database.schemas import *


async def create_user(db: Session, user: HashedUser):
    user = Users(**user.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


async def create_user_external_map(db: Session, user_external_map: UserExternalMap):
    user_external_map = UserExternalMapping(**user_external_map.model_dump())
    db.add(user_external_map)
    db.commit()
    db.refresh(user_external_map)
    return user_external_map


async def create_post(db: Session, post: PostForm, user_id: str):
    post = Posts(**post.model_dump(), author_id=user_id)
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


async def create_comment(db: Session, comment: CommentForm, user_id: str):
    comment = Comments(**comment.model_dump(), author_id=user_id)
    comment.post.increment_comment_count()
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


async def create_like(db: Session, author_id: str, post_id: int):
    like = Likes(author_id=author_id, post_id=post_id)
    like.post.increment_like_count()
    db.add(like)
    db.commit()
    db.refresh(like)
    return like


async def read_user_by_id(db: Session, user_id: str):
    return db.query(Users).filter(Users.user_id == user_id).first()


async def read_user_by_email(db: Session, email: str):
    return db.query(Users).filter(Users.email == email).first()


async def read_user_external_map(db: Session, external_id: str, provider: str):
    return (
        db.query(UserExternalMapping)
        .filter(
            and_(
                UserExternalMapping.external_id == external_id,
                UserExternalMapping.provider == provider,
            )
        )
        .first()
    )


async def read_post(db: Session, post_id: int):
    return db.query(Posts).filter(Posts.post_id == post_id).first()


async def read_whole_post(db: Session, post_id: int):
    anchor_clause = db.query(Comments)
    anchor_clause = anchor_clause.filter(Comments.parent_comment_id == None)
    anchor_clause = anchor_clause.cte('all_comments', recursive=True)

    recursive_clause = db.query(Comments)
    recursive_clause = recursive_clause.join(anchor_clause, Comments.parent_comment_id == anchor_clause.c.comment_id)

    recursive_cte = anchor_clause.union_all(recursive_clause)

    post = (
        db.query(Posts)
        .filter(Posts.post_id == post_id)
        .outerjoin(recursive_cte, recursive_cte.c.post_id == Posts.post_id)
        .options(selectinload(Posts.comments))
        .first()
    )
    if post:
        post.increment_view_count()
    db.commit()

    return post


async def read_comment(db: Session, comment_id: int):
    return db.query(Comments).filter(Comments.comment_id == comment_id).first()


async def search_posts(
    db: Session, category: str = None, author_id: str = None, keyword: str = None
):
    query = db.query(Posts)

    if category:
        query = query.filter(Posts.category == category)

    if author_id:
        query = query.filter(Posts.author_id == author_id)

    if keyword:
        query = query.filter(
            or_(
                Posts.title.ilike(f"%{keyword}%"),
                Posts.content.ilike(f"%{keyword}%"),
            )
        )

    return query.all()


async def search_comment(db: Session, author_id: str = None, post_id: int = None):
    query = db.query(Comments)

    if post_id:
        query = query.filter(Comments.post_id == post_id)

    if author_id:
        query = query.filter(Comments.author_id == author_id)

    return query.all()


async def read_like(db: Session, author_id: str, post_id: int):
    return (
        db.query(Likes)
        .filter(Likes.author_id == author_id, Likes.post_id == post_id)
        .all()
    )


async def create_image(db: Session, image: Image):
    image = Images(**image.model_dump())
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


async def delete_post(db: Session, post: Posts):
    db.delete(post)
    db.commit()


async def delete_comment(db: Session, comment: Comments):
    comment.post.decrement_comment_count()
    
    parent_comment = comment.parent_comment
    if parent_comment:
        parent_comment.decrement_child_comment_count()
        if parent_comment.child_comment_count == 0:
            db.delete(parent_comment)
        else:
            db.delete(comment)
    else:
        if comment.child_comment_count == 0:
            db.delete(comment)
        else:
            comment.author_id = None
            comment.content = "삭제된 댓글입니다"
    db.commit()


async def delete_comment_content(db: Session, comment: Comments):
    comment.post.decrement_comment_count()
    comment.author_id = None
    comment.content = "삭제된 댓글입니다."
    db.commit()