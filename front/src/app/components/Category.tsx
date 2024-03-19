"use client";
import { useRouter, usePathname } from "next/navigation";
import { useSession } from "next-auth/react";

const Category = () => {
  const router = useRouter();
  const { data: session } = useSession();
  const pathname = usePathname();

  const HomePostClick = () => {
    router.push("/Community");
  };

  const FreePostClick = () => {
    router.push("/Community/FreePost");
  };

  const PopularityPostClick = () => {
    router.push("/Community/PopularityPost");
  };

  const SellPostClick = () => {
    router.push("/Community/SellPost");
  };

  const BuyPostClick = () => {
    router.push("/Community/BuyPost");
  };

  const PostCreateBt = async () => {
    try {
      const response = await fetch(`${process.env.Localhost}/post/temp`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      router.push(`/Community/PostCreate/${data.temp_post_id}`);
    } catch (error) {
      console.error("Error", error);
    }
  };

  return (
    <main className="flex w-full h-full justify-center items-center">
      <div className="flex pl-1 space-x-4 bg-[#FFFFFF] h-[40px] w-[90%]">
        <div
          className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${
            pathname === "/Community"
              ? "text-[#F4A460] border-b border-[#F4A460]"
              : "text-[#808080]"
          }`}
          onClick={HomePostClick}
        >
          홈
        </div>
        <div
          className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${
            pathname === "/Community/FreePost"
              ? "text-[#F4A460] border-b border-[#F4A460]"
              : "text-[#808080]"
          }`}
          onClick={FreePostClick}
        >
          자유
        </div>
        <div
          className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${
            pathname === "/Community/PopularityPost"
              ? "text-[#F4A460] border-b border-[#F4A460]"
              : "text-[#808080]"
          }`}
          onClick={PopularityPostClick}
        >
          인기
        </div>
        <div
          className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${
            pathname === "/Community/SellPost"
              ? "text-[#F4A460] border-b border-[#F4A460]"
              : "text-[#808080]"
          }`}
          onClick={SellPostClick}
        >
          팝니다
        </div>
        <div
          className={`flex items-center cursor-pointer hover:text-[#F4A460] hover:border-b-2 border-[#F4A460] h-[40px] ${
            pathname === "/Community/BuyPost"
              ? "text-[#F4A460] border-b border-[#F4A460]"
              : "text-[#808080]"
          }`}
          onClick={BuyPostClick}
        >
          삽니다
        </div>
      </div>
      <button
        className={`items-center w-[80px] h-[30px] bg-blue-500 hover:bg-blue-700 text-white font-bold rounded`}
        onClick={PostCreateBt}
      >
        글쓰기
      </button>
    </main>
  );
};

export default Category;
