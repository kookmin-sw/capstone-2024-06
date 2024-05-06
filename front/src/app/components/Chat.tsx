"use client";
import { useState, MouseEventHandler, useEffect, useRef } from "react";
import Image from "next/image";
import { useSession } from "next-auth/react";

const ChatModal = ({
  onClose,
}: {
  onClose: MouseEventHandler<HTMLButtonElement>;
}) => {
  const ChatScrollRef = useRef<HTMLDivElement>(null);
  const ScrollToBottom = () => {
    if (ChatScrollRef.current) {
      ChatScrollRef.current.scrollIntoView({ block: "end" });
    }
  };

  const { data: session } = useSession();
  const [ChatDatas, SetChatDatas] = useState([
    {
      last_chat: {
        chat_history_id: 0,
        created_at: "",
        message: "",
        receiver_id: "",
        sender_id: "",
      },
      opponent: {
        email: "",
        image: "",
        name: "",
        user_id: "",
      },
    },
  ]);
  const [ChatingData, SetChatingData] = useState([
    {
      chat_history_id: 0,
      created_at: "",
      message: "",
      receiver_id: "",
      sender_id: "",
    },
  ]);
  const EnterKey = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      SendMessage();
    }
  };

  const [Message, SetMessage] = useState("");
  const MessageChange = (e: any) => {
    SetMessage(e.target.value);
  };

  const [Client, SetClient] = useState<WebSocket | null>(null);
  const OpenWebSoket = (UserId: string) => {
    const newClient = new WebSocket(
      `ws://${process.env.OnlyiP}/chat/${UserId}`
    );
    newClient.onopen = () => {
      newClient.send((session as any)?.access_token);
    };
    newClient.onmessage = (message) => {
      const parsed = JSON.parse(message.data);
      AddMessage(parsed.message, parsed.sender_id);
    };
    SetClient(newClient);
  };

  const AddMessage = (message: string, id: string) => {
    const newMessage = {
      chat_history_id: 0,
      created_at: "",
      message: message,
      receiver_id: "",
      sender_id: id,
    };
    SetChatingData((PrevChatData) => [...PrevChatData, newMessage]);
  };

  const SendMessage = () => {
    if (Message.trim() !== "" && Client) {
      Client.send(Message);
      AddMessage(Message, "");
      SetMessage("");
    }
  };

  const CloseWebSocket = () => {
    if (Client) {
      Client.close();
      SetClient(null);
    }
  };

  const [OpenChatings, SetOpenChatings] = useState(true);
  const [OpenChat, SetOpenChat] = useState(false);
  const [AnotherProfile, SetAnotherProfile] = useState("");
  const [AnotherName, SetAnotherName] = useState("");

  const OpenChatWindow = (
    Id: string,
    chat_history_id: number,
    anotherprofile: string,
    anothername: string,
    anotheruserid: string
  ) => {
    if (Id != "") {
      SetAnotherProfile(anotherprofile);
      SetAnotherName(anothername);
      OpenWebSoket(anotheruserid);
      const ChatingLoad = async () => {
        try {
          const response = await fetch(
            `${process.env.Localhost}/chat/history/${Id}?last_chat_history_id=${
              chat_history_id + 1
            }`,
            {
              method: "GET",
              headers: {
                Authorization: `Bearer ${(session as any)?.access_token}`,
                "Content-Type": "application/json",
              },
            }
          );
          const data = await response.json();
          SetChatingData(data);
        } catch (error) {
          console.error("Error", error);
        }
      };
      ChatingLoad();
    } else {
      SetChatingData([]);
      SetOpenChatings(!OpenChatings);
      CloseWebSocket();
    }
    SetOpenChat(!OpenChat);
  };
  useEffect(() => {
      ScrollToBottom();
  }, [ChatingData]);

  useEffect(() => {
    if (!session) return;
    const ChatLoad = async () => {
      try {
        const response = await fetch(`${process.env.Localhost}/chat/room`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        });
        const data = await response.json();
        SetChatDatas(data);
      } catch (error) {
        console.error("Error", error);
      }
    };
    ChatLoad();
  }, [OpenChatings]);

  return (
    <div className="fixed inset-0 z-50 overflow-auto bg-opacity-75 flex">
      <div className="absolute right-0 bottom-0 p-4 w-[400px]">
        <div className="relative bg-white shadow-lg rounded-lg text-gray-900 ">
          <div className="flex justify-between items-center p-2 border-b bg-gradient-to-r from-cyan-500 to-blue-500 rounded-t-lg">
            <div className="text-xl font-semibold text-[#FFFAFA] ml-2">
              Direct Message
            </div>
            <button
              className="text-[#FFFAFA] text-3xl  focus:outline-none mb-2"
              onClick={onClose}
            >
              &times;
            </button>
          </div>
          {OpenChat && (
            <div className="p-4 bg-[#F2F2F2] h-[520px] overflow-y-auto">
              {ChatingData.map((ChatData, index) => (
                <div key={index} ref={ChatScrollRef}>
                  {ChatData.sender_id != (session as any)?.user.user_id &&
                    ChatData.sender_id != "" && (
                      <div className="p-2 w-full h-fit mb-1">
                        <div className="flex">
                          <div className="w-[50px] mr-2">
                            <Image
                              src={AnotherProfile}
                              alt="Profile image"
                              width={100}
                              height={1}
                              objectFit="cover"
                              className="rounded-full"
                            />
                          </div>
                          <div className="flex-col w-full">
                            <div className="text-sm text-gray-900 dark:text-white mb-1">
                              {AnotherName}
                            </div>
                            <div className="flex flex-col w-fit h-fit border bg-white rounded-lg p-2 ">
                              {ChatData.message}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                  {(ChatData.sender_id === (session as any)?.user.user_id ||
                    ChatData.sender_id === "") && (
                    <div className="flex justify-end items-end w-full">
                      <div className="w-fit h-fit bg-[#F7D358] rounded-lg p-2 mb-3">
                        {ChatData.message}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {!OpenChat && (
            <div className="p-4 bg-[#F2F2F2] h-[520px] overflow-y-auto">
              {ChatDatas.map((ChatData, index) => (
                <div
                  key={index}
                  className="p-2 w-full h-fit bg-white hover:bg-gray-300 cursor-pointer rounded-lg mb-2"
                  onClick={() =>
                    OpenChatWindow(
                      ChatData.opponent.user_id,
                      ChatData.last_chat.chat_history_id,
                      ChatData.opponent.image,
                      ChatData.opponent.name,
                      ChatData.opponent.user_id
                    )
                  }
                >
                  <div className="flex">
                    <div className="w-[50px] mr-2">
                      <Image
                        src={ChatData.opponent.image}
                        alt="Profile image"
                        width={100}
                        height={100}
                        objectFit="cover"
                        className="border-black rounded-full"
                      />
                    </div>
                    <div className="flex-col w-full">
                      <div className="text-base text-black dark:text-white mb-1">
                        {ChatData.opponent.name}
                      </div>
                      <div className="text-sm text-gray-400 dark:text-white ">
                        {ChatData.last_chat.message}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="flex  bottom-0 left-0 right-0 p-4 bg-white rounded-b-lg">
            <input
              type="text"
              value={Message}
              onChange={MessageChange}
              onKeyDown={(e) => EnterKey(e)}
              placeholder="메세지를 입력하세요..."
              className="w-3/4 border border-gray-300 rounded px-3 py-2"
            />
            <button
              onClick={SendMessage}
              className="w-1/4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              전송
            </button>
            <button onClick={() => OpenChatWindow("", 0, "", "", "")}>
              임시
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const Chat = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const ModalClick = () => {
    setIsModalOpen(!isModalOpen);
  };

  return (
    <div>
      <button
        onClick={ModalClick}
        className="text-sm text-[#808080] mx-1 cursor-pointer"
      >
        채팅
      </button>
      {isModalOpen && <ChatModal onClose={ModalClick} />}
    </div>
  );
};

export default Chat;
