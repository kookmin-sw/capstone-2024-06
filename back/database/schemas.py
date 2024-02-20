from pydantic import BaseModel


class User(BaseModel):
    username: str
    password: str


class Post(BaseModel):
    post_id: str
    title: str
    content: str
    category: str
    writer_username: str


class Comment(BaseModel):
    comment_id: str
    content: str
    writer_username: str
    post_id: str