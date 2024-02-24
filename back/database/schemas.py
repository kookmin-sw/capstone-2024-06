from pydantic import BaseModel


class User(BaseModel):
    user_id: str
    password: str


class BasePost(BaseModel):
    author_id: str
    title: str
    category: str


class PostForm(BasePost):
    content: str


class PostPreview(BasePost):
    post_id: int
    like_count: int
    view_count: int


class Post(PostPreview):
    content: str


class Comment(BaseModel):
    author_id: str
    post_id: int
    content: str


class Image(BaseModel):
    image_id: str
    filename: str