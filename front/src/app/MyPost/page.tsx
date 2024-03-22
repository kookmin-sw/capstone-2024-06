"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Image from "next/image";
import Nav from "../components/Nav";
import MyPageProfile from "../components/MyPageProfile";

const MyPost = () => {
  const { data: session } = useSession();
  const router = useRouter();

  return (
    <>
      <Nav />
      <MyPageProfile />
      <div className="absolute left-[647px] top-[149px] container mx-auto p-4">
        <div className="absolute w-[532px] h-[182px] left-[30px] top-[60px] border border-gray-300" />
        <div className="absolute left-[50px] top-[70px] font-semibold text-base leading-9 text-black">
          제목
        </div>
        <div className="absolute left-[50px] top-[100px] font-thin text-lg leading-[39px] text-black">
          내용을 여기에 추가하세요
        </div>
        <div className="absolute left-[60px] top-[200px] ">
          <Image
            src="/Heart.png"
            alt="Heart image"
            width={20}
            height={20}
            className="cursor-pointer mr-1 rounded-full"
          />
        </div>
        <div className="absolute left-[90px] top-[200px] ">
          <Image
            src="/Comments.png"
            alt="Comments image"
            width={22}
            height={22}
            className="cursor-pointer mr-1 rounded-full"
          />
        </div>
      </div >
    </>
  );
};

export default MyPost;
