"use client";
import { useState, ChangeEvent, DragEvent } from "react";
import { useSession } from "next-auth/react";
import Image from "next/image";
import RecommendImgSlider from "./RecommendImgSlider";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";

interface ImagePreview {
  url: string;
  file: File;
}

const AnalysisImageUpLoader = () => {
  const { data: session } = useSession();

  const [images, setImages] = useState<string[]>([]);
  const [plotlyHTML, setPlotlyHTML] = useState('');
  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null);

  // 버퍼링
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // 분석하기 버튼을 눌렀을 때
  const [AnalyBtClick, SetAnalyBtClick] = useState(true);
  const AnalyBtClicks = () => {
    SetAnalyBtClick(!AnalyBtClick);
    console.log(AnalyBtClick);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImagePreview({ url, file });
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      const url = URL.createObjectURL(file);
      setImagePreview({ url, file });
    }
  };

  const ImageDeleteBt = () => {
    setImagePreview(null);
  };

  const ImageAnalysisBt = async () => {
    try {
      if (imagePreview) {
        setIsAnalyzing(true);
        const formData = new FormData();
        formData.append("file", imagePreview.file);
        const ImagePost = await fetch(
          `${process.env.Localhost}/prototype_process`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
            body: formData,
          }
        );
        const ImageDatas = await ImagePost.json();
        console.log(ImageDatas);
        setImages(ImageDatas.file_name);
        setPlotlyHTML(ImageDatas.plot);
        setIsAnalyzing(false);
        AnalyBtClicks();
      }
    } catch (error) {
      console.error("Error", error);
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="my-10">
      {AnalyBtClick && (
        <div className="flex flex-col items-center justify-center mt-4">
          <div
            className="m-2 relative"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {imagePreview ? (
              <img
                src={imagePreview.url}
                alt="Image preview"
                className="w-[500px] h-[400px] cursor-pointer"
              />
            ) : (
              <div className="flex my-4 items-center justify-center border-dashed border-2 text-[#808080] text-sm w-[600px] h-[400px] cursor-pointer">
                드래그하여 사진 업로드
              </div>
            )}
          </div>
          {!imagePreview ? (
            <div className="flex">
              <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileChange}
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
              >
                파일 선택
              </label>
            </div>
          ) : (
            <div className="flex">
              {isAnalyzing ? (
                <button
                  type="button"
                  className="ml-1 cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded"
                  disabled
                >
                  <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                </button>
              ) : (
                <div className="flex">
                  <div
                    className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
                    onClick={ImageDeleteBt}
                  >
                    사진제거
                  </div>
                  <div
                    className="ml-1 cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
                    onClick={ImageAnalysisBt}
                  >
                    분석하기
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      {!AnalyBtClick && <RecommendImgSlider Images={images} />}
      <iframe className="plot" srcDoc={plotlyHTML} width="1000" height="800"></iframe>
    </main>
  );
};

export default AnalysisImageUpLoader;
