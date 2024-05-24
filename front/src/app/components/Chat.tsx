"use client";
import {
  useState,
  MouseEventHandler,
  useEffect,
  useRef,
  ChangeEvent,
} from "react";
import Image from "next/image";
import { useSession } from "next-auth/react";
import { useSearchParams } from "next/navigation";

const ChatModal = ({
  onClose,
  First,
}: {
  onClose: MouseEventHandler<HTMLButtonElement>;
  First: boolean;
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
        image: { filename: "", image_id: "" },
        sender_id: "",
      },
      opponent: {
        email: "",
        image: "",
        name: "",
        user_id: "",
      },
      unread: true,
    },
  ]);

  const [UserData, SetUserData] = useState({
    email: "",
    followed: false,
    followee_count: 0,
    follower_count: 0,
    image: "",
    name: "‍",
    user_id: "",
  });

  const params = useSearchParams();
  const user_id = params.get("user_id");

  const [ChatingData, SetChatingData] = useState([
    {
      chat_history_id: 0,
      created_at: "",
      image: { filename: "", image_id: "" },
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

  const ESCKey = async (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.keyCode === 27) {
      OpenChatWindow("", 0, "", "", "");
      console.log("ESC");
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
      newClient.send(
        JSON.stringify({
          type: "token",
          content: (session as any)?.access_token,
        })
      );
    };
    newClient.onmessage = (message) => {
      const parsed = JSON.parse(message.data);
      const pparse = JSON.parse(parsed);
      const images_ids = pparse.image ? pparse.image.image_id : null;
      AddMessage(pparse.message, pparse.sender_id, images_ids);
    };
    SetClient(newClient);
  };

  const AddMessage = (message: string, id: string, image_id: string) => {
    const newMessage = {
      chat_history_id: 0,
      created_at: "",
      message: message,
      receiver_id: "",
      image: { filename: "", image_id: image_id },
      sender_id: id,
    };
    SetChatingData((PrevChatData) => [...PrevChatData, newMessage]);
  };

  const SendMessage = () => {
    if (Message.trim() !== "" && Client) {
      Client.send(JSON.stringify({ type: "text", content: Message }));
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

  const OpenFirstWindow = (
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
            `/what-desk-api/chat/history/${Id}`,
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
            `/what-desk-api/chat/history/${Id}?last_chat_history_id=${
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
          console.log(data);
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
        const response = await fetch(`/what-desk-api/chat/room`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        });

        [
          {
            type: "missing",
            loc: ["query", "last_chat_history_id"],
            msg: "Field required",
            input: null,
            url: "https://errors.pydantic.dev/2.6/v/missing",
          },
        ];
        const data = await response.json();
        console.log(data);
        SetChatDatas(data);
      } catch (error) {
        console.error("Error", error);
      }
    };
    ChatLoad();
  }, [OpenChatings]);

  useEffect(() => {
    if (First === false) {
      First = true;
      SetOpenChat(!OpenChat);
      const fetchUserData = async () => {
        try {
          const response = await fetch(
            `/what-desk-api/user/profile/${user_id}`,
            {
              method: "GET",
              headers: {
                Authorization: `Bearer ${(session as any)?.access_token}`,
                "Content-Type": "application/json",
              },
            }
          );
          if (response.ok) {
            const userData = await response.json();

            SetUserData(userData);
            OpenFirstWindow(
              userData.user_id,
              -1,
              userData.image,
              userData.name,
              userData.user_id
            );
          } else {
            console.error("Failed to fetch user data");
          }
        } catch (error) {
          console.error("Error fetching user data:", error);
        }
      };
      fetchUserData();
    }
  }, []);

  const [ChatImage, SetChatImage] = useState<File | null>(null);
  var url: any;

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      SetChatImage(file);
      url = URL.createObjectURL(file);
      console.log(url);
    }
  };

  const ChatImageSend = (file: any) => {
    console.log(url);
    if (Client && Client.readyState === WebSocket.OPEN) {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        if (typeof reader.result === "string") {
          const base64Content = reader.result?.split(",")[1];
          Client.send(
            JSON.stringify({
              type: "image",
              filename: file.name,
              content: base64Content,
            })
          );
        }
      };
    }
    // AddMessage("", session?.user.user_id , url)
    SetChatImage(null);
  };

  return (
    <div
      className="fixed inset-0 z-50 overflow-auto bg-opacity-75 flex "
      tabIndex={0}
      onKeyDown={(e) => ESCKey(e)}
    >
      <div className="absolute right-0 bottom-0 p-4 w-[400px]">
        <div className="relative bg-white shadow-lg rounded-lg text-gray-900 ">
          <div className="flex justify-between items-center p-2 border-b bg-gradient-to-r from-cyan-500 to-blue-500 rounded-t-lg">
            <div className="text-xl font-semibold text-[#FFFAFA] ml-2">
              Direct Message
            </div>
            <button
              className="text-[#FFFAFA] text-3xl  focus:outline-none mb-2"
              onClick={(e) => {
                onClose(e);
                CloseWebSocket();
              }}
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
                            {ChatData.message ? (
                              <div className="flex flex-col w-fit h-fit border bg-white rounded-lg p-2 ">
                                {ChatData.message}
                              </div>
                            ) : ChatData.image.image_id ? (
                              <div className="flex w-[100px] ">
                                <Image
                                  src={`${process.env.Localhost}${ChatData.image.image_id}?w=300&h=300`}
                                  alt={ChatData.image.filename}
                                  width={100}
                                  height={100}
                                  objectFit="cover"
                                  className="rounded-md"
                                />
                              </div>
                            ) : (
                              <div className="flex flex-col w-fit h-fit border bg-white rounded-lg p-2 ">
                                {ChatData.message}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                  {(ChatData.sender_id === (session as any)?.user.user_id ||
                    ChatData.sender_id === "") && (
                    <div className="flex justify-end items-end w-full">
                      {ChatData.message ? (
                        <div className="w-fit h-fit bg-[#F7D358] rounded-lg p-2 mb-3">
                          {ChatData.message}
                        </div>
                      ) : (
                        <div className="flex w-[140px] border">
                          <Image
                            src={`${process.env.Localhost}${ChatData.image.image_id}?w=300&h=300`}
                            alt={ChatData.image.filename}
                            width={300}
                            height={300}
                            objectFit="cover"
                            className="rounded-md"
                          />
                        </div>
                      )}
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
                      {ChatData.last_chat.message ? (
                        <div className="text-sm text-gray-400 dark:text-white ">
                          {ChatData.last_chat.message}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-400 dark:text-white ">
                          사진을 보냈습니다.
                        </div>
                      )}
                    </div>
                    {!ChatData.unread ? (
                      <></>
                    ) : (
                      <div className="h-[50px] flex items-center">
                        <div className="w-[15px]">
                          <Image
                            src="/alret.PNG"
                            alt="Profile image"
                            width={100}
                            height={1}
                            objectFit="cover"
                            className="rounded-full"
                          />
                        </div>
                      </div>
                    )}
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
              className="w-3/4 border border-gray-300 rounded px-3 py-2 mr-2"
            />

            {!ChatImage ? (
              <div className="flex">
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  onChange={handleFileChange}
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer w-[110px]  h-full text-white bg-blue-700 hover:bg-blue-800 font-medium rounded text-sm px-5 py-2.5 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800"
                >
                  파일 선택
                </label>
              </div>
            ) : (
              <div
                className="w-[110px] h-full text-white bg-blue-700 hover:bg-blue-800 font-medium rounded text-sm px-5 py-2.5 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800"
                onClick={() => ChatImageSend(ChatImage)}
              >
                사진 전송
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const Chat = ({ First }: { First: boolean }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const ModalClick = () => {
    setIsModalOpen(!isModalOpen);
  };

  return First ? (
    <div>
      <button
        onClick={ModalClick}
        className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
      >
        채팅
      </button>
      {isModalOpen && <ChatModal onClose={ModalClick} First={true} />}
    </div>
  ) : (
    <div>
      <button
        onClick={ModalClick}
        className="bg-blue-500 text-white rounded-md px-2 py-1 cursor-pointer ml-3"
      >
        메세지 보내기
      </button>
      {isModalOpen && <ChatModal onClose={ModalClick} First={false} />}
    </div>
  );
};

export default Chat;
