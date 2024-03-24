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
          분석 결과
        </div>
        <div className="absolute w-[1049px] h-0 left-[179px] top-[184px] border border-gray-300 transform rotate-0.05" />

      </div>
    </>
  );
};

export default EditPost;
