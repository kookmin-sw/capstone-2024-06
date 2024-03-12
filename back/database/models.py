from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, Sequence, DateTime
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.orm import relationship
from database.database import Base
from datetime import datetime


class Follows(Base):
    __tablename__ = "follows"

    follower_user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    followee_user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)


class Users(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)

    name = Column(String)
    email = Column(String, unique=True)
    image = Column(String)
    hashed_password = Column(String)

    posts = relationship("Posts", back_populates="author", uselist=True)
    comments = relationship("Comments", back_populates="author", uselist=True)
    liked_posts = relationship(
        "Posts",
        secondary="post_likes",
        back_populates="liking_users",
        cascade="all, delete",
        uselist=True,
    )
    liked_comments = relationship(
        "Comments",
        secondary="comment_likes",
        back_populates="liking_users",
        cascade="all, delete",
        uselist=True,
    )
    user_external_map = relationship("UserExternalMapping")
    followers = relationship(
        "Users",
        secondary="follows",
        primaryjoin=user_id == Follows.followee_user_id,
        secondaryjoin=user_id == Follows.follower_user_id,
        backref="followees",
    )


class Posts(Base):
    __tablename__ = "posts"

    post_id = Column(Integer, Sequence("post_id_seq"), primary_key=True)

    author_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    category = Column(String, nullable=False)

    like_count = Column(Integer, default=0, nullable=False)
    view_count = Column(Integer, default=0, nullable=False)
    comment_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    author = relationship("Users", back_populates="posts")
    comments = relationship(
        "Comments",
        primaryjoin="and_(Posts.post_id == Comments.post_id, Comments.parent_comment_id == None)",
        back_populates="post",
        cascade="all, delete-orphan",
        uselist=True,
    )
    liking_users = relationship(
        "Users",
        secondary="post_likes",
        back_populates="liked_posts",
        cascade="all, delete",
        uselist=True,
    )
    images = relationship("Images", cascade="all, delete-orphan")

    @hybrid_method
    def increment_view_count(self):
        self.view_count += 1

    @hybrid_method
    def increment_like_count(self):
        self.like_count += 1

    @hybrid_method
    def increment_comment_count(self):
        self.comment_count += 1

    @hybrid_method
    def decrement_comment_count(self):
        self.comment_count -= 1


class Comments(Base):
    __tablename__ = "comments"

    comment_id = Column(Integer, Sequence("comment_id_seq"), primary_key=True)

    author_id = Column(String, ForeignKey("users.user_id"))
    post_id = Column(Integer, ForeignKey("posts.post_id"), nullable=False)
    parent_comment_id = Column(Integer, ForeignKey("comments.comment_id"))

    content = Column(String, nullable=False)
    like_count = Column(Integer, default=0, nullable=False)
    child_comment_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    author = relationship("Users", back_populates="comments")
    post = relationship("Posts", back_populates="comments")
    parent_comment = relationship(
        "Comments",
        remote_side=[comment_id],
        back_populates="child_comments",
    )
    child_comments = relationship(
        "Comments",
        back_populates="parent_comment",
        uselist=True,
        cascade="all, delete-orphan",
    )
    liking_users = relationship(
        "Users",
        secondary="comment_likes",
        back_populates="liked_comments",
        cascade="all, delete",
        uselist=True,
    )

    @hybrid_method
    def increment_child_comment_count(self):
        self.child_comment_count += 1

    @hybrid_method
    def decrement_child_comment_count(self):
        self.child_comment_count -= 1

    @hybrid_method
    def increment_like_count(self):
        self.like_count += 1


class PostLikes(Base):
    __tablename__ = "post_likes"

    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.post_id"), primary_key=True)


class CommentLikes(Base):
    __tablename__ = "comment_likes"

    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    comment_id = Column(Integer, ForeignKey("comments.comment_id"), primary_key=True)


class TempPosts(Base):
    __tablename__ = "temp_posts"

    temp_post_id = Column(Integer, Sequence("temp_post_id_seq"), primary_key=True)

    author_id = Column(String, ForeignKey("users.user_id"), unique=True)

    images = relationship("Images", cascade="all, delete-orphan")
    author = relationship("Users")


class Images(Base):
    __tablename__ = "images"

    image_id = Column(String, primary_key=True)

    temp_post_id = Column(Integer, ForeignKey("temp_posts.temp_post_id"))
    post_id = Column(Integer, ForeignKey("posts.post_id"))
    filename = Column(String, nullable=False)


class UserExternalMapping(Base):
    __tablename__ = "user_external_mapping"

    external_id = Column(String, primary_key=True)
    provider = Column(String, primary_key=True)

    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    user = relationship("Users", back_populates="user_external_map")
