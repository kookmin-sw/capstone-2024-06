import Nav from "@/app/components/Nav";
import Category from "@/app/components/Category";
import Posts from "@/app/components/Posts"

export default function FreePost() {
  return (
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex border-b justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <Category />
        </div>
      </div>
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="w-full">
            <Posts PostCateGory="자유"/>
          </div>
        </div>
      </div>
    </main>
  );
}
