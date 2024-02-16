import PictureUpload from "./components/PictureUpload";
import Nav from "./components/Nav";
import ImageSlider from "./components/ImageSlider";
import RecommendImgSlider from "./components/RecommendImgSlider";

export default function Home() {
  return (
    <main className="flex-col justify-center w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center border min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          {/* <div className="w-full my-5 border">
            <ImageSlider />
          </div> */}
          <div className="my-5 flex-col">
            <div className="font-bold text-lg">잘꾸민 책상 모음</div>
            <RecommendImgSlider />
          </div>
        </div>
      </div>
    </main>
  );
}
