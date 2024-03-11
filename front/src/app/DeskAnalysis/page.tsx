import Nav from "../components/Nav";
import PictureUpload from "../components/PictureUpload";

export default function DeskAnalysis() {
  return (
    <main className="flex-col justify-center w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="my-5">
            <PictureUpload />
          </div>
        </div>
      </div>
    </main>
  );
}
