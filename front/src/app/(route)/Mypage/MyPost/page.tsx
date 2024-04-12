"use client";
import Nav from "../../../components/Nav";
import MyPosting from "../../../components/MyPosting";

const MyPost = () => {


  return (
    <>
      <main className="flex-col w-full h-full">
        <Nav />
        <div className="relative w-screen h-screen bg-[background-color]">
          <div className="absolute w-[251px] h-[91px] left-[200px] top-[66px] font-inter font-semibold text-4xl leading-14 text-yellow-600">
            어떤데스크
          </div>
          <div className="absolute w-[173px] h-[83px] left-[200px] top-[144px] font-semibold text-base leading-9 text-black">
            내가 쓴 글
          </div>
          <div className="absolute w-[1049px] h-0 left-[179px] top-[184px] border border-gray-300 transform rotate-0.05">
          </div>
          <div className="flex justify-center w-full h-auto">
            <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto relative">
              <div className="w-full absolute top-[200px] left-[50px]">
                <MyPosting />
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
};

export default MyPost;