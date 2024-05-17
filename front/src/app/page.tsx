import Nav from "./components/Nav";
import Image from "next/image";
import Link from "next/link";
import RecommendImgSlider from "./components/RecommendImgSlider";


export default function Home() {

  return (
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="flex-wrap w-full h-[900px] justify-center items-center">
            <div className="flex items-center justify-center w-full text-8xl h-1/2  font-extrabold">
              Imagine Your Desk
            </div>
            <div className="flex items-center justify-center w-full text-5xl font-semibold">
              당신의 책상을 상상해보세요
            </div>
          </div>
          <div className="flex w-full h-auto">
            <div className="w-1/2 h-fit sticky top-5">
              <div className="flex-col space-y-10">
                <div className="text-4xl font-bold">
                  오늘은 이런 책상 어떤가요?
                </div>
                <div className="text-xl font-semibold">
                  최상의 옵션으로 준비했습니다..
                  <br></br>완성은 언제 할 수 있을까요..
                </div>
              </div>
            </div>
            <div className="w-1/2 h-full ml-auto overflow-y-auto">
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/desk4.jpg"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/desk4.jpg"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/desk4.jpg"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/desk4.jpg"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
            </div>
          </div>
          <div className="w-full flex justify-center mt-20">
            <button className="animate-bounce bg-transparent w-[200px] h-[50px] hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded">
              <Link href={"/Main"}>시작하기</Link>
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
