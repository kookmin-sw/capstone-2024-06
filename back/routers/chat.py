from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

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
    token = await websocket.receive_text()
    user_id = get_current_user(token)

    if user_id in connections:
        del connections[user_id]
    connections[user_id] = websocket
    print(f"user <{user_id}> connected")

    try:
        while True:
            message = await websocket.receive_text()
            chat_history = BaseChatHistory(
                message=message, sender_id=user_id, receiver_id=opponent_id
            )
            await crud.create_chat_history(db, chat_history)
            if opponent_id in connections:
                await connections[opponent_id].send_text(chat_history.model_dump_json())

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
