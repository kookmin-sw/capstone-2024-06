"use client";
import { useRouter, usePathname } from "next/navigation";

const Category = () => {
  const router = useRouter();
  const pathname = usePathname();

  const HomePostClick = () => {
    router.push("/Community")
  }

  const FreePostClick = () => {
    router.push("/Community/FreePost");
  };

  const SellPostClick = () => {
    router.push("/Community");
  };

  const BuyPostClick = () => {
    router.push("/Community");
  };

  return (
    <main className="flex w-full h-full">
      <div className="flex pl-1 space-x-4 bg-[#FFFFFF] h-[40px]">
        <div className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${pathname === "/Community" ? "text-[#F4A460] border-b border-[#F4A460]" :"text-[#808080]"}`} onClick={HomePostClick}>홈</div>
        <div className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${pathname === "/Community/FreePost" ? "text-[#F4A460] border-b border-[#F4A460]" :"text-[#808080]"}`} onClick={FreePostClick}>
          자유게시판
        </div>
        <div className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px]`} onClick={SellPostClick}>
          팝니다
        </div>
        <div className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px]`} onClick={BuyPostClick}>
          삽니다
        </div>
        <div className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px]`}>좀더 추가</div>
      </div>
    </main>
  );
};

export default Category;
