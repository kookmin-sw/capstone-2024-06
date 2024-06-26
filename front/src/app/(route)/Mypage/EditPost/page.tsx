"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Image from "next/image";
import Nav from "../../../components/Nav";
import MyPageProfile from "../../../components/MyPageProfile";

const EditPost = () => {
  const { data: session } = useSession();
  const router = useRouter();


  return (
    <>
      <Nav />
      <div className="relative w-screen h-screen bg-[background-color]">

        <div className="absolute w-[251px] h-[91px] left-[200px] top-[66px] font-inter font-semibold text-4xl leading-14 text-yellow-600">
          어떤데스크
        </div>
        <div className="absolute w-[173px] h-[83px] left-[200px] top-[144px] font-semibold text-base leading-9 text-black">
          임시저장
        </div>
        <div className="absolute w-[1049px] h-0 left-[179px] top-[184px] border border-gray-300 transform rotate-0.05" />
        <div className="absolute box-border w-[1000px] h-[413px] left-[202px] top-[224px] border  border-transparent border-opacity-0">
          <div className="absolute left-6 top-2 font-semibold text-base leading-9 text-black">
            <label htmlFor="title" className="sr-only">제목</label>
            <input
              id="title"
              name="title"
              type="text"
              className="w-[940px] border border-gray-300 px-2 py-1 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="제목을 입력하세요"
            />
          </div>

          <div className="absolute w-[950px] h-0 left-[20px] top-[60px] border border-solid border-gray-300"></div>
          <input
            type="text"
            className="absolute left-[25px] top-[70px] w-[940px] h-[290px] p-2 border border-gray-300 rounded"
            placeholder="내용을 여기에 추가하세요"
          />

          <div className="absolute w-[1000px] h-0 left-[0px] top-[370px] border border-solid border-gray-400"></div>
          <div className="absolute left-[930px] top-[380px]  font-actor font-normal text-base leading-38 text-center text-gray-400">저장 | 3</div>
        </div>
      </div>
    </>
  );
};

export default EditPost;
