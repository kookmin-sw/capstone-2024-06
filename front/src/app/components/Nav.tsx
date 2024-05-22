"use client";
import Image from "next/image";
import { useRouter, usePathname } from "next/navigation";
import { SetStateAction, useState } from "react";
import { useSession, signOut } from "next-auth/react";
import Chat from "./Chat";

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
  };

  const EnterKey = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      router.push(`/Community?keyword=${SearchValue}`);
      SetSearchValue("")
    }
  };

  return (
    <div className="flex justify-center w-full h-[70px] border-b">
      <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12">
        <div className="mr-6 w-[120px]">
          <Image
            src="/logo.png"
            alt="name"
            width={50}
            height={100}
            onClick={LogoImgClick}
            className="cursor-pointer"
          />
        </div>
        <div className="font-mono  w-[280px] text-semibold font-[600] flex space-x-4">
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
        <div className="flex justify-center items-center">
          <div className="relative w-[300px] border rounded-lg mr-3">
            <input
              type="search"
              className="block w-full p-3  text-sm text-gray-900"
              placeholder="검색 내용을 입력하세요 !"
              value={SearchValue}
              onChange={SearchValueChange}
              onKeyDown={(e) => EnterKey(e)}
            />
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
          </div>
        )}
        {session && session.user &&(
          <div className="flex justify-center items-center w-fit">
            <div className="text-sm text-[#808080] mx-1 cursor-pointer">
              {session.user.name?.substring(0,4)}
            </div>
            <div
              className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
              onClick={MyPageClick}
            >
              마이페이지
            </div>
            <button
              className="text-sm text-[#808080] mx-1 cursor-pointer hover:text-[#F4A460]"
              onClick={() => signOut()}
            >
              로그아웃
            </button>
            <Chat First={true}/>
          </div>
        )}
      </div>
    </div>
  );
};

export default Nav;
