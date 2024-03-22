import Nav from "@/app/components/Nav";
import PostCreates from "@/app/components/PostCreates";

export default function PostCreate() {
  return (
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <PostCreates />
        </div>
      </div>
    </main>
  );
}
