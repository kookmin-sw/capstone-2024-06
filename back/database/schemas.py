from pydantic import BaseModel


class User(BaseModel):
    username: str


class UserSignUp(User):
    password: str


class UserDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None

