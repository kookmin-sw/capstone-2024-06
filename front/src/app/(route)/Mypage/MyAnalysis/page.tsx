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
        <div className="absolute w-[300px] h-[200px] left-[220px] top-[208px] bg-gray-300"></div>
        <div className="absolute w-[564px] h-[104px] left-[220px] top-[430px] font-actor font-normal text-black text-l leading-[48px]">
          {session?.user?.name}님의 사진은 ... 느낌입니다
        </div>
        <div className="absolute w-[912px] h-[110px] left-[600px] top-[257px] font-actor font-normal text-black text-l leading-[48px]">
          원하는 사진의 개수를 선택하세요
        </div>
        <div className="absolute flex items-center space-x-4 left-[600px] top-[301px]">
          <input
            type="checkbox"
            id="option1"
            name="option1"
            value="1"
            className="form-checkbox h-5 w-5 text-yellow-600"
          />
          <label htmlFor="option1">1</label>
        </div>
        <div className="absolute flex items-center space-x-4 left-[650px] top-[301px]">
          <input
            type="checkbox"
            id="option2"
            name="option2"
            value="2"
            className="form-checkbox h-5 w-5 text-yellow-600"
          />
          <label htmlFor="option2">2</label>
        </div>
        <div className="absolute flex items-center space-x-4 left-[700px] top-[301px]">
          <input
            type="checkbox"
            id="option3"
            name="option3"
            value="3"
            className="form-checkbox h-5 w-5 text-yellow-600"
          />
          <label htmlFor="option3">3</label>
        </div>
        <div className="absolute flex items-center space-x-4 left-[750px] top-[301px]">
          <input
            type="checkbox"
            id="option4"
            name="option4"
            value="4"
            className="form-checkbox h-5 w-5 text-yellow-600"
          />
          <label htmlFor="option4">4</label>
        </div>
        <div className="absolute w-[300px] h-[400px] left-[600px] top-[381px] bg-gray-300"></div>
        <div className="absolute w-[300px] h-[400px] left-[908px] top-[381px] bg-gray-300"></div>
      </div>
    </>
  );

};

export default EditPost;
