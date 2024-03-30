"use client";
import { useState, MouseEventHandler } from "react";
import Message from "./Message";
import Image from "next/image";

const ChatModal = ({
  onClose,
}: {
  onClose: MouseEventHandler<HTMLButtonElement>;
}) => {
  return (
    <div className="fixed inset-0 z-50 overflow-auto bg-opacity-75 flex">
      <div className="absolute right-0 bottom-0 p-4 w-[400px]">
        <div className="relative bg-white shadow-lg rounded-lg text-gray-900">
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
          <div className="p-4 bg-[#F2F2F2] max-h-[600px] min-h-[600px] overflow-y-auto">
          <div className="p-2 w-full h-fit bg-white hover:bg-gray-300 cursor-pointer rounded-lg">
              <div className="flex">
                <div className="w-[50px] mr-2">
                  <Image
                    src="/Profilex2.webp"
                    alt="Profile image"
                    width={100}
                    height={100}
                    objectFit="cover"
                    className="border-black rounded-full"
                  />
                </div>
                <div className="flex-col w-full">
                  <div className="text-base text-black dark:text-white mb-1">박근우</div>
                  <div className="text-sm text-gray-400 dark:text-white ">내용</div>
                </div>
              </div>
            </div>
            <div className="p-2 w-full h-[80px]">
              <div className="flex">
                <div className="w-[50px]">
                  <Image
                    src="/Profilex2.webp"
                    alt="Profile image"
                    width={100}
                    height={1}
                    objectFit="cover"
                    className="border-black rounded-full"
                  />
                </div>
                <div className="flex-col w-full">
                  <div className="text-sm text-gray-900 dark:text-white mb-1">박근우</div>
                  <div className="flex flex-col max-w-full h-full border bg-white rounded-lg"></div>
                </div>
              </div>
            </div>
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
