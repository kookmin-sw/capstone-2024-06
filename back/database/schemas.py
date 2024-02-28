from pydantic import BaseModel
from typing import List


class User(BaseModel):
    user_id: str
    password: str


class HashedUser(BaseModel):
    user_id: str
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class CommentForm(BaseModel):
    post_id: int
    content: str


class Comment(CommentForm):
    comment_id: int
    author_id: str


class BasePost(BaseModel):
    title: str
    category: str


class PostForm(BasePost):
    content: str


class PostPreview(BasePost):
    author_id: str
    post_id: int
    like_count: int
    view_count: int


class Post(PostPreview):
    content: str
    comments: List[Comment]


class Image(BaseModel):
    image_id: str
    filename: str