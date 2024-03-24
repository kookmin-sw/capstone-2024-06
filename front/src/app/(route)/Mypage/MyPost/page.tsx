"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Image from "next/image";
import Nav from "../../../components/Nav";
import MyPageProfile from "../../../components/MyPageProfile";

const MyPost = () => {
  const { data: session } = useSession();
  const router = useRouter();

  const EditpostClick = () => {
    router.push("/Mypage/EditPost");
  }

  return (
    <>
      <Nav />
      <div className="relative w-screen h-screen bg-[background-color]">

        <div className="absolute w-[251px] h-[91px] left-[200px] top-[66px] font-inter font-semibold text-4xl leading-14 text-yellow-600">
          어떤데스크
        </div>
        <div className="absolute w-[173px] h-[83px] left-[200px] top-[144px] font-semibold text-base leading-9 text-black">
          작성 완료
        </div>
        <div className="relative left-[1090px] top-[144px]">
          <button
            onClick={EditpostClick}
            className="flex w-[100px] h-[33px] justify-center rounded-md bg-yellow-600 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-yellow-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
          >
            임시 저장
          </button>
        </div>



        <div className="absolute w-[1049px] h-0 left-[179px] top-[184px] border border-gray-300 transform rotate-0.05" />
        <div className="absolute w-[1032px] h-[182px] left-[189px] top-[224px] border border-gray-300" />
        <div className="absolute left-[206px] top-[235px] font-semibold text-base leading-9 text-black">
          제목
        </div>
        <div className="absolute w-[1205px] h-[173px] left-[209px] top-[285px] font-thin text-lg leading-[39px] text-black">
          내용을 여기에 추가하세요
        </div>
        <div className="absolute left-[206px] top-[355px] ">
          <Image
            src="/Heart.png"
            alt="Heart image"
            width={20}
            height={20}
            className="cursor-pointer mr-1 rounded-full"
          />
        </div>
        <div className="absolute left-[236px] top-[355px] ">
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
