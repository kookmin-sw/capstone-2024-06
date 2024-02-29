from sqlalchemy import Column, String, Integer, ForeignKey, Sequence
from sqlalchemy.orm import relationship
from database.database import Base


class Users(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)

    name = Column(String)
    email = Column(String)
    image = Column(String)
    provider = Column(String)
    hashed_password = Column(String)

    posts = relationship("Posts", back_populates="author", uselist=True)
    comments = relationship("Comments", back_populates="author", uselist=True)
    likes = relationship("Likes", back_populates="author", uselist=True)


class Posts(Base):
    __tablename__ = "posts"

    post_id = Column(Integer, Sequence("post_id_seq"), primary_key=True)

    author_id = Column(String, ForeignKey("users.user_id"))

    title = Column(String)
    content = Column(String)
    category = Column(String)

    like_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)

    author = relationship("Users", back_populates="posts")
    comments = relationship("Comments", back_populates="post", uselist=True)
    likes = relationship("Likes", back_populates="post", uselist=True)


class Comments(Base):
    __tablename__ = "comments"

    comment_id = Column(Integer, Sequence("comment_id_seq"), primary_key=True)

    author_id = Column(String, ForeignKey("users.user_id"))
    post_id = Column(Integer, ForeignKey("posts.post_id"))

    content = Column(String)

    author = relationship("Users", back_populates="comments")
    post = relationship("Posts", back_populates="comments")


class Likes(Base):
    __tablename__ = "likes"
    
    author_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.post_id"), primary_key=True)

    author = relationship("Users", back_populates="likes")
    post = relationship("Posts", back_populates="likes")


class Images(Base):
    __tablename__ = "images"

    image_id = Column(String, primary_key=True)
    filename = Column(String)