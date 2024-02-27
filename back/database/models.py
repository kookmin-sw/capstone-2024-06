from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database.database import Base


class User(Base):
    __tablename__ = 'users'

    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)
    