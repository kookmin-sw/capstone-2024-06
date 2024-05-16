import os
import shutil
import uuid

from fastapi import APIRouter, Depends, UploadFile

from sqlalchemy.orm import Session

from database import crud
from database.models import *
from database.schemas import *
from dependencies import *
from config_loader import config


router = APIRouter(
    prefix="/community",
    tags=["community"]
)


@router.get("/post/temp", response_model=TempPost)
async def create_temporary_code(
    user_id: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    temp_post = await crud.create_temp_post(db, user_id)
    return temp_post


@router.post("/post/{temp_post_id}")
async def create_post(
    temp_post_id: int,
    post: PostForm,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    await crud.create_post(db, post, user_id, temp_post_id)
    return {"message": "Post created successfully"}


@router.get("/post/search", response_model=list[PostPreview])
async def search_posts(
    category: str | None = None,
    author_id: str | None = None,
    keyword: str | None = None,
    order: str = "newest",
    per: int = 24,
    page: int = 1,
    user_id: str | None = Depends(get_current_user_if_signed_in),
    db: Session = Depends(get_db),
):
    if order not in ["newest", "most_viewed", "most_scrroutered", "most_liked"]:
        return HTTPException(status_code=400, detail="Invalid order parameter")

    posts = await crud.search_posts(
        db,
        category=category,
        author_id=author_id,
        keyword=keyword,
        order=order,
        per=per,
        page=page,
        user_id=user_id,
    )
    return posts


@router.get("/post/{post_id}", response_model=Post)
async def read_post(
    post_id: int,
    user_id: str | None = Depends(get_current_user_if_signed_in),
    db: Session = Depends(get_db),
):
    post = await crud.read_post_with_view(db, post_id, user_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post does not exist.")
    return post


@router.get("/comment/{post_id}", response_model=list[Comment])
async def read_comment(post_id: int, db: Session = Depends(get_db)):
    comments = await crud.read_comments(db, post_id)
    return comments


@router.delete("/post/{post_id}")
async def delete_post(
    post_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if post.author_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Permission denied: You are not the author of this post",
        )

    await crud.delete_post(db, post)
    return {"message": "Post deleted successfully"}


@router.post("/post/scrap/{post_id}")
async def scrap_post(
    post_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    scrap = await crud.read_post_scrap(db, user_id, post_id)
    if scrap is None:
        await crud.create_post_scrap(db, user_id, post_id)
        return {"message": "User scrapped post successfully"}
    else:
        await crud.delete_post_scrap(db, user_id, post_id)
        return {"message": "User unscrapped post successfully"}
    

@router.post("/post/like/{post_id}")
async def scrap_post(
    post_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    like = await crud.read_post_like(db, user_id, post_id)
    if like is None:
        await crud.create_post_like(db, user_id, post_id)
        return {"message": "User liked post successfully"}
    else:
        await crud.delete_post_like(db, user_id, post_id)
        return {"message": "User unliked post successfully"}
    


@router.post("/comment/like/{comment_id}")
async def scrap_comment(
    comment_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    comment = await crud.read_comment(db, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    comment_like = await crud.read_comment_like(db, user_id, comment_id)
    if comment_like is None:
        await crud.create_comment_like(db, user_id, comment_id)
        return {"message": "User liked comment successfully"}
    else:
        return {"message": "User alread liked comment"}


@router.post("/comment")
async def create_comment(
    comment: CommentForm,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = await crud.read_post(db, comment.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    await crud.create_comment(db, comment, user_id)
    return {"messaeg": "Comment created successfully"}


@router.delete("/comment/{comment_id}")
async def delete_comment(
    comment_id: int,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    comment = await crud.read_comment(db, comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    if not comment.author_id:
        raise HTTPException(status_code=404, detail="Comment already deleted")

    if comment.author_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="You are not the author of this comment",
        )

    await crud.delete_comment(db, comment)
    return {"message": "Comment deleted successfully"}


@router.post("/image/{temp_post_id}", response_model=Image)
async def upload_image(
    temp_post_id: int,
    file: UploadFile,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    upload_path = config["PATH"]["upload"]
    filename = file.filename
    file_extension = os.path.splitext(filename)[1]
    file_path = os.path.join(upload_path, str(uuid.uuid4()) + file_extension)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_id = "/" + file_path
    image = Image(image_id=image_id, filename=filename)
    return await crud.create_image(db, image, temp_post_id)


@router.post("/follow/{followee_user_id}")
async def follow_user(
    followee_user_id: str,
    follower_user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    await crud.create_follow(db, follower_user_id, followee_user_id)
    return {"message": "Followed successfully"}


@router.delete("/follow/{followee_user_id}")
async def unfollow_user(
    followee_user_id: str,
    follower_user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    follow = await crud.read_follow(db, follower_user_id, followee_user_id)
    if not follow:
        raise HTTPException(status_code=404, detail="You are not following this user")

    await crud.delete_follow(db, follow)
    return {"message": "Unfollowed successfully"}