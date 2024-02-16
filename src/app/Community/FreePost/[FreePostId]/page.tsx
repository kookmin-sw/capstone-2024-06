import Nav from "@/app/components/Nav";
import Image from "next/image";
import Comment from "@/app/components/Comment";

export default function FreePost() {
  
  const PostDummyData = {
    PostImage: "/desk5.png",
    PostId: 1,
    PostTitle: "예시 입니다",
    PostContent: "이 문구는 예시로 작성되어있습니다.",
  };

  return (
    <main className="flex-col justify-center w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto mt-4">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="w-full h-auto">
            <div className="w-full border-b flex">
              <div className="text-xl font-bold pl-3 w-full">
                {PostDummyData.PostTitle}
              </div>
              <div className="mr-2">2024.02.18</div>
              <div className="mr-2">19:31</div>
            </div>
            <div className="w-full border-b flex items-center">
              <Image src="/Profilex.png" width={40} height={30} alt={""} />
              <div className="ml-2 w-full">user name</div>
              <div className="flex w-[100px] h-full space-x-1 mr-1">
                <div>조회수</div>
                <div>22</div>
              </div>
              <div className="flex w-[100px] h-full space-x-1,mr-1">
                <div>좋아요</div>
                <div>22</div>
              </div>
              <div className="flex w-[100px] h-full space-x-1">
                <div>댓글</div>
                <div>22</div>
              </div>
            </div>
            <div className="flex w-full justify-center items-center my-10">
              <Image
                src={PostDummyData.PostImage}
                width={600}
                height={300}
                alt={""}
              />
            </div>
            <div className="flex justify-start items-center text-xl">{PostDummyData.PostContent}</div>
          </div>
          <div className="border w-full h-auto">
            <Comment />
          </div>
        </div>
      </div>
    </main>
  );
}
