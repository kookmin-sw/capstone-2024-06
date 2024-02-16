"use client";
import Image from "next/image";
import { useRouter, usePathname } from "next/navigation";

const Nav = () => {
  const router = useRouter();
  const pathname = usePathname();

  const LogoImgClick = () => {
    router.push("/");
  };

  const CommunityClick = () => {
    router.push("/Community");
  };

  const DeskAnalysisClick = () => {
    router.push("/DeskAnalysis");
  };

  const MyPageClick = () => {
    router.push("/MyPage");
  };

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
            className={`cursor-pointer hover:text-[#F4A460] ${
              pathname.startsWith("/Community") ? "text-[#F4A460]" : "text-[#808080]"
            }`}
            onClick={CommunityClick}
          >
            커뮤니티
          </div>
          <div
            className={`cursor-pointer hover:text-[#F4A460] ${
              pathname === "/DeskAnalysis" ? "text-[#F4A460]" : "text-[#808080]"
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
              />
            </div>
          </div>
        </div>
        <div className="flex justify-center w-1/3">
          <div className="text-sm text-[#808080] mx-1 cursor-pointer">
            로그인
          </div>
          <div className="text-sm text-[#808080] mx-1 cursor-pointer">
            회원가입
          </div>
          <div className="text-sm text-[#808080] mx-1 cursor-pointer" onClick={MyPageClick}>
            마이페이지
          </div>
        </div>
      </div>
    </div>
  );
};

export default Nav;
