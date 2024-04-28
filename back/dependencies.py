import base64
import json
from jose import jwt
from passlib.context import CryptContext

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from database.database import SessionLocal


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

SECRET_KEY = "secret"  # temp
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def decode_jwt_payload(token):
    header, payload, signature = token.split(".")

    decoded_payload = base64.urlsafe_b64decode(payload + "==").decode("utf-8")
    parsed_payload = json.loads(decoded_payload)

    return parsed_payload


def get_current_user(token: str = Depends(oauth2_scheme)):
    # try:
    #     payload = decode_jwt_payload(token)
    #     jwt.decode(token, SECRET_KEY, ALGORITHM)
    #     return payload["sub"]
    # except:
    #     raise HTTPException(status_code=401, detail="Invalid token")
    return "admin"


def get_current_user_if_signed_in(token: str | None = Depends(optional_oauth2_scheme)):
    # try:
    #     if not token or token == "undefined":
    #         return None

    #     payload = decode_jwt_payload(token)
    #     jwt.decode(token, SECRET_KEY, ALGORITHM)
    #     return payload["sub"]
    # except:
    #     raise HTTPException(status_code=401, detail="Invalid token")
    return "admin"