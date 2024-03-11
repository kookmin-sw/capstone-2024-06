import Nav from "@/app/components/Nav";
import Image from "next/image";
import Comment from "@/app/components/Comment";
import Posting from "@/app/components/Posting";

export default function FreePost() {
  const PostDummyData = {
    PostImage: "/desk5.png",
    PostId: 1,
    PostTitle: "어떤 데스크 ",
    PostContent:
      "이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. 이 문구는 예시로 gggg 작성되어있습니다. 이 문구는 예시로 작성되어있습니다. ",
  };

  return (
    <main className="flex-col justify-center w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto mt-4">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <Posting />
        </div>
      </div>
    </main>
  );
}
