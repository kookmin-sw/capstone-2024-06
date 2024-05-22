from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import os
import uuid

from sqlalchemy.orm import Session

from database import crud
from database.models import *
from database.schemas import *
from dependencies import *


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


connections = dict()


@router.websocket("/{opponent_id}")
async def chatting_websocket(
    opponent_id: str, websocket: WebSocket, db: Session = Depends(get_db)
):
    await websocket.accept()
    data = await websocket.receive_json()
    if data["type"] != "token":
        HTTPException(status_code=401, detail="Should pass token first")
    user_id = get_current_user(data["content"])

    if user_id in connections:
        del connections[user_id]
    connections[user_id] = websocket
    print(f"user <{user_id}> connected")

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "text":
                message = data["content"]
                chat_history = BaseChatHistory(
                    message=message, sender_id=user_id, receiver_id=opponent_id
                )

            elif data["type"] == "image":
                image_data = base64.b64decode(data["content"])
                filename = data["filename"]
                filename = "test.jpg"
                upload_path = config["PATH"]["upload"]
                file_extension = os.path.splitext(filename)[1]
                file_path = os.path.join(upload_path, str(uuid.uuid4()) + file_extension)

                with open(file_path, "wb") as f:
                    f.write(image_data)

                image_id = "/" + file_path
                image = Image(image_id=image_id, filename=filename)
                await crud.create_chat_image(db, image)

                chat_history = BaseChatHistory(
                    image_id=image_id, sender_id=user_id, receiver_id=opponent_id
                )

            chat_history = await crud.create_chat_history(db, chat_history)
            chat_history = ChatHistory.model_validate(chat_history)

            await connections[user_id].send_json(chat_history.model_dump_json())
            if opponent_id in connections:
                await connections[opponent_id].send_json(chat_history.model_dump_json())

    except WebSocketDisconnect:
        ...
    finally:
        print(f"user <{user_id}> disconnected")
        if user_id in connections:
            del connections[user_id]
        connections[user_id] = websocket
        await crud.update_chat_access_history(db, user_id, opponent_id)


@router.get("/history/{opponent_id}", response_model=list[ChatHistory])
async def get_chat_histories(
    opponent_id: str,
    last_chat_history_id: int | None = None,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat_histories = await crud.read_chat_histories(
        db, user_id, opponent_id, last_chat_history_id
    )
    return chat_histories


@router.get("/room", response_model=list[ChatRoom])
async def get_chatting_rooms(user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    chat_rooms = await crud.read_chatting_rooms(db, user_id)
    return chat_rooms
