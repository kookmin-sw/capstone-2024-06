import Nav from "@/app/components/Nav";
import Posting from "@/app/components/Posting";

export default function SellPost() {
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
