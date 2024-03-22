"use client";
import Image from "next/image";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";

const MyPageProfile = () => {
  const { data: session } = useSession();

  const router = useRouter();

  const [MyPageProfileIcons, SetMyPageProfileIcons] = useState([
    "/Write.png",
    "/Heart.png",
    "/NotFollow.png",
    "/Chat.png",
  ]);
  const MyPageProfileName = ["글쓰기", "좋아요", "팔로우", "채팅"]

  const PostCreateClick = () => {
    router.push("/Community/PostCreate")
  }

  const MyPostClick = () => {
    router.push("/MyPost")
  }

  const HeartClick = async () => {

  }
  return (
    <main className="w-[600px] mt-10 border h-[400px]">
      <div className="absolute w-[173px] h-[83px] left-[697px] top-[99px] font-semibold text-base leading-9 text-black hover:underline"
        onClick={MyPostClick}>
        내가 쓴 글
      </div>
      <div className="absolute w-[173px] h-[83px] left-[807px] top-[99px] font-semibold text-base leading-9 text-black hover:underline">
        댓글 단 글
      </div>
      <div className="absolute w-[173px] h-[83px] left-[917px] top-[99px] font-semibold text-base leading-9 text-black hover:underline">
        분석
      </div>
      <div className="absolute w-[500px] h-[0px] left-[644px] top-[151px] border border-gray-300 transform rotate-0.05"></div>
      {session && (
        <div className="flex-col">
          <div className="flex w-full h-[150px]">
            <div className="flex h-full w-1/2 justify-center items-center">
              <div className="w-[100px] h-[100px]">
                <Image
                  src={session.user?.image ?? ""}
                  //src="/Profilex2.webp"
                  alt="Profile image"
                  width={1000}
                  height={1000}
                  objectFit="cover"
                  className="cursor-pointer mr-1 rounded-full"
                />
              </div>
            </div>
            <div className="flex w-1/2 h-full items-center">
              <div className="text-2xl">{session.user?.name}</div>
            </div>
          </div>
          <div className="flex justify-center items-center w-full h-[60px]">
            <div className="w-[250px] bg-[#ced4da] font-bold rounded-sm h-[30px] flex items-center justify-center">
              프로필 수정
            </div>
          </div>
          <div className="flex justify-center h-[60px]">
            <div className="mr-1">팔로잉</div>
            <div className="mr-1">0</div>
            <div className="mr-1">팔로워</div>
            <div>0</div>
          </div>
          <div className="flex w-full h-[100px] justify-center items-center space-x-5">
            {MyPageProfileIcons.map((icon, index) => (
              <div key={index}>
                <div>
                  <Image
                    src={icon}
                    alt="Profile image"
                    width={40}
                    height={40}
                    objectFit="cover"
                    className="cursor-pointer mr-1"
                    onClick={PostCreateClick}
                  />
                </div>
                <div>{MyPageProfileName[index]}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
};

export default MyPageProfile;