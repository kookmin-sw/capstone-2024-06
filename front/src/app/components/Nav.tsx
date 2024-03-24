"use client";
import type { Metadata } from 'next'
import Image from "next/image";
import { useRouter, usePathname } from "next/navigation";
import { SetStateAction, useState } from "react";
import { useSession, signOut } from "next-auth/react";



const Nav = () => {
  const { data: session } = useSession();

  const router = useRouter();
  const pathname = usePathname();

  const [SearchValue, SetSearchValue] = useState("");

  const SearchValueChange = (e: {
    target: { value: SetStateAction<string> };
  }) => {
    SetSearchValue(e.target.value);
  };

  const LogoImgClick = () => {
    router.push("/");
  };

  const LoginClick = () => {
    router.push("/login/sign-in");
  };

  const CommunityClick = () => {
    router.push("/Community");
  };

  const DeskAnalysisClick = () => {
    router.push("/DeskAnalysis");
  };

  const MyPageClick = () => {
    router.push("/Mypage");
  };

  const SignupClick = () => {
    router.push("/login/sign-up");
  }

  console.log(session);

  return (
    <div className="flex justify-center w-full h-[70px] border-b">
      <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12">
        <div className="mr-6 w-[120px]">
          <Image
            src="/namet.png"
            alt="name"
            width={100}
            height={27}
            onClick={LogoImgClick}
            className="cursor-pointer"
          />
        </div>
        <div className="font-mono w-1/4 text-semibold font-[600] flex space-x-4">
          <div
            className={`cursor-pointer hover:text-[#F4A460] ${pathname.startsWith("/Community")
              ? "text-[#F4A460]"
              : "text-[#808080]"
              }`}
            onClick={CommunityClick}
          >
            커뮤니티
          </div>
          <div
            className={`cursor-pointer hover:text-[#F4A460] ${pathname === "/DeskAnalysis" ? "text-[#F4A460]" : "text-[#808080]"
              }`}
            onClick={DeskAnalysisClick}
          >
            책상분석
          </div>
        </div>
        <div className="flex justify-center">
          <div className="xl:w-96">
            <div className="relativ w-full">
              <input
                type="search"
                className="w-[300px] px-3 h-[40px] text-gray-700 bg-white border border-solid border-gray-300 rounded-md focus:text-gray-700 focus:bg-white focus:border-blue-600 focus:outline-none"
                placeholder="검색하기"
                value={SearchValue}
                onChange={SearchValueChange}
              />
            </div>
          </div>
        </div>
        {!session && (
          <div className="flex justify-center w-1/3">
            <div
              className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
              onClick={LoginClick}
            >
              로그인
            </div>
            <div
              className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
              onClick={SignupClick}
            >
              회원가입
            </div>
            {/* <div
              className="text-sm text-[#808080] mx-1 cursor-pointer"
              onClick={MyPageClick}
            >
              마이페이지
            </div> */}
          </div>
        )}
        {session && (
          <div className="flex justify-center w-1/3">
            <div className="text-sm text-[#808080] mx-1 cursor-pointer">
              {session.user?.name}
            </div>
            <div
              className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
              onClick={MyPageClick}
            >
              마이페이지
            </div>
            <button
              className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
              onClick={() => signOut()}>로그아웃</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Nav;
