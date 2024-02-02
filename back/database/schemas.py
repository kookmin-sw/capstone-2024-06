from pydantic import BaseModel


class UserBase(BaseModel):
    id: str
    password: str