import Nav from "./components/Nav";
import Image from "next/image";
import Link from "next/link";



export default function Home() {
  const DeskSampleImage = ['/sample1.PNG','/sample2.PNG','/sample3.PNG','/sample4.PNG']
  return (
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="flex-wrap w-full h-[900px] justify-center items-center">
            <div className="flex items-center justify-center w-full text-7xl h-1/2  font-extrabold">
              어떤 데스크
            </div>
            <div className="flex items-center justify-center w-full text-3xl font-semibold">
              당신의 완벽한 책상을 찾아 드립니다
            </div>
          </div>
          <div className="flex w-full h-auto">
            <div className="w-1/2 h-fit sticky top-5">
              <div className="flex-col space-y-10">
                <div className="text-2xl font-bold">
                어떤 책상이 당신의 스타일에 맞을까요?
                </div>
                <div className="text-xl font-semibold">
                당신의 업무 효율을 높여줄 책상 추천
                  
                </div>
              </div>
            </div>
            <div className="w-1/2 h-full ml-auto overflow-y-auto">
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/sample1.PNG"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/sample2.PNG"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/sample3.PNG"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
              <div className="border-none h-fit rounded-3xl overflow-hidden mb-2">
                <Image
                  src="/sample4.PNG"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                />
              </div>
            </div>
          </div>
          <div className="w-full flex justify-center my-20">
            <button className="animate-bounce bg-transparent w-[200px] h-[50px] hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded">
              <Link href={"/Main"}>시작하기</Link>
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
