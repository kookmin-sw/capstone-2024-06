"use client";
import { useState, useRef } from "react";
import { useSession } from "next-auth/react";
import Image from "next/image";
import Nav from "../../../components/Nav";
import SettingForm from "../../../components/settingUserinfo";

interface IProps {
  profileImg?: any;
}

const EditPost = () => {
  const { data: session } = useSession();

  return (
    <>
      <Nav />
      <div className="relative w-screen h-screen bg-[background-color]">
        <div className="absolute w-[251px] h-[91px] left-[200px] top-[66px] font-inter font-semibold text-4xl leading-14 text-yellow-600">
          어떤데스크
        </div>
        <div className="absolute w-[173px] h-[83px] left-[200px] top-[144px] font-semibold text-base leading-9 text-black">
          프로필 수정
        </div>
        <div className="absolute w-[1049px] h-0 left-[179px] top-[184px] border border-gray-300 transform rotate-0.05" />
      </div>
      <div className="flex h-full w-1/2 justify-center items-center">
        <div className="absolute w-[200px] h-[200px] left-[220px] top-[384px]">
          <Image
            src={session?.user?.image ?? ""}
            alt="Profile image"
            width={200}
            height={200}
            objectFit="cover"
            className="cursor-pointer mr-1 rounded-full"
          />
        </div>
      </div>
      <div className="absolute w-[173px] h-[83px] left-[260px] top-[600px] font-semibold text-base leading-9 text-black">
        프로필 사진 변경
      </div>
      <SettingForm user={session?.user} />
    </>
  );
};

export default EditPost;
