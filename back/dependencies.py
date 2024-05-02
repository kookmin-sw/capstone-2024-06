import base64
import json
from jose import jwt
from passlib.context import CryptContext

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from database.database import SessionLocal
from config_loader import config


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


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
    try:
        payload = decode_jwt_payload(token)
        jwt.decode(token, config["TOKEN"]["secret"], config["TOKEN"]["algorithm"])
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_if_signed_in(token: str | None = Depends(optional_oauth2_scheme)):
    try:
        if not token or token == "undefined":
            return None

        payload = decode_jwt_payload(token)
        jwt.decode(token, config["TOKEN"]["secret"], config["TOKEN"]["algorithm"])
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")