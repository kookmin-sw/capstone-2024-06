from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from database.database import Base


class User(Base):
    __tablename__ = 'user'

    username = Column(String, primary_key=True, index=True)
    password = Column(String)
    post = relationship("Post", back_populates="writer", uselist=True)
    comment = relationship("Comment", back_populates="writer", uselist=True)


class Post(Base):
    __tablename__ = 'post'
    
    post_id = Column(String, primary_key=True)
    title = Column(String)
    content = Column(String)
    category = Column(String)
    writer_username = Column(String, ForeignKey("user.username"))
    writer = relationship("User", back_populates="post")
    comment = relationship("Comment", back_populates="post", uselist=True)


class Comment(Base):
    __tablename__ = 'comment'

    comment_id = Column(String, primary_key=True)
    content = Column(String)
    writer_username = Column(String, ForeignKey("user.username"))
    writer = relationship("User", back_populates="comment")
    post_id = Column(String, ForeignKey("post.post_id"))
    post = relationship("Post", back_populates="comment")
