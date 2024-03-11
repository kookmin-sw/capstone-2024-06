import Nav from "@/app/components/Nav";
import Category from "@/app/components/Category";

export default function FreePost() {
  return (
    <main className="flex-col justify-center border-b w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="w-full">
            <Category />
          </div>
        </div>
      </div>
    </main>
  );
}
