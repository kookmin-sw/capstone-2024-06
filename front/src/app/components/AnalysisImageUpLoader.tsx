"use client";
import { useState, ChangeEvent, DragEvent, useEffect, useMemo } from "react";
import { useSession } from "next-auth/react";
import Image from "next/image";
import RecommendImgSlider from "./RecommendImgSlider";
import ImageAnalysisSlider from "./ImageAnalysisSlider";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";


interface ImagePreview {
  url: string;
  file: File;
}

const AnalysisImageUpLoader = () => {
  const [UploadImageBt, SetUploadImageBt] = useState(false);
  const [RecommendImageBt, SetRecommendImageBt] = useState(false);
  const { data: session } = useSession();

  const [images, setImages] = useState<string[]>([]);
  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null);

  // 버퍼링

  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // 분석하기 버튼을 눌렀을 때
  const [AnalyBtClick, SetAnalyBtClick] = useState(false);

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

  const [RecommendImageGet, SetRecommendImageGet] = useState(false);
  const [AnalysisImageGet, SetAnalysisImageGet] = useState(false);

  const [AnalysisImage, SetAnalysisImage] = useState([
    { index: 0, src_url: "", landing: "" },
  ]);

  const ImageAnalysisBt = async () => {
    try {
      if (imagePreview) {
        setIsAnalyzing(true);
        const formData = new FormData();
        formData.append("file", imagePreview.file);
        const ImagePost = await fetch(
          `/api/recommend/image`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
            body: formData,
          }
        );
        const ImageDatas = await ImagePost.json();
        SetAnalysisImage(ImageDatas);
        SetAnalysisImageGet(true);
        setImages(ImageDatas.file_name);
        setIsAnalyzing(false);
        SetAnalyBtClick(true);
    
      }
    } catch (error) {
      console.error("Error", error);
      setIsAnalyzing(false);
    }
  };

  const [SampleImage, SetSampleImage] = useState([
    {
      index: 0,
      src_url: "",
      landing: "",
      score: 1,
    },
  ]);
  
  useEffect(() => {
    const SampleImageGet = async () => {
      try {
        const ImagePost = await fetch(
          `/api/recommend/sample`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
          }
        );
        const SampleImageDatas = await ImagePost.json();

        SetSampleImage(SampleImageDatas);
      } catch (error) {
        console.error("Error", error);
      }
    };
    SampleImageGet();
  }, []);

  const [RecommendImage, SetRecommendImage] = useState([
    { index: 0, src_url: "", landing: "" },
  ]);

  const SampleImageScoreSend = async () => {
    try {
      const ScoreData = SampleImage.map((data) => ({
        index: data.index,
        rating: data.score,
      }));
      const ImagePost = await fetch(
        `/api/recommend/preference`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(ScoreData),
        }
      );
      const SampleImageScoreSendData = await ImagePost.json();
      SetRecommendImage(SampleImageScoreSendData);
      SetRecommendImageGet(true);
    } catch (error) {
      console.error("Error", error);
    }
  };

  const ScrollDown = () => {
    const scrollPosition = 2000;
    window.scrollTo({
      top: scrollPosition,
      behavior: "smooth",
    });
  };

  const ImageUploadImageClick = () => {
    SetUploadImageBt(true);
    SetRecommendImageBt(false);
    ScrollDown();
  };

  const RecommendImageClick = () => {
    SetUploadImageBt(false);
    SetRecommendImageBt(true);
    ScrollDown();
  };

  return (
    <main className="my-10">
      <div className="flex-col h-[600px]">
        <div className="flex items-center justify-center h-1/3">
          <div className="text-2xl font-semibold">
            책상 분석 어떤 것이 더 좋으세요 ?
          </div>
        </div>
        <div className="flex items-center justify-center w-full h-2/3">
          <div className="flex-col items-center justify-center w-1/3 ">
            <div className="flex items-center justify-center w-full">
              <div className="w-[120px] mb-4">
                <Image
                  src="/imageupload.png"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                  className="cursor-pointer hover:scale-105"
                  onClick={ImageUploadImageClick}
                />
              </div>
            </div>
            <div className="text-sm flex items-center justify-center text-sm font-semibold">
              자신의 이미지 업로드 하여 추천 받기
            </div>
          </div>
          <div className="flex-col items-center justify-center w-1/3">
            <div className="flex items-center justify-center w-full">
              <div className="w-[300px] mb-4">
                <Image
                  src="/star.png"
                  alt="Post Image"
                  width={1000}
                  height={1000}
                  style={{ width: "100%", height: "auto" }}
                  className="cursor-pointer hover:scale-105"
                  onClick={RecommendImageClick}
                />
              </div>
            </div>
            <div className="text-sm flex items-center justify-center text-sm font-semibold">
              이미지들을 받아 평가하고 추천받기
            </div>
          </div>
        </div>
      </div>
      {UploadImageBt && (
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
              <div className="flex my-4 items-center justify-center border-dashed border-4 text-[#808080] text-sm w-[800px] h-[400px] cursor-pointer">
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
                  분석중
                  <FontAwesomeIcon
                    icon={faSpinner}
                    className="animate-spin ml-2"
                  />
                </button>
              ) : !AnalyBtClick && (
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
          {AnalyBtClick && (
            <div className="flex-col mt-10">
              <div className="w-full flex justify-center mb-5 font-semibold text-xl h-12 border-b-2">사진 분석 결과</div>
              <ImageAnalysisSlider Images={AnalysisImage} />
            </div>
          )}
        </div>
      )}
      {RecommendImageBt && (
        <div>
          {(
            <div>
              <div className="flex mt-10">
                <RecommendImgSlider Images={SampleImage} />
              </div>
              {!RecommendImageGet && (<div className="flex justify-center items-center mt-10 w-full">
                <button
                  className="bg-transparent hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
                  onClick={SampleImageScoreSend}
                >
                  제출하기
                </button>
              </div>)}
            </div>
          )}
          <div className="flex mt-10">
            {RecommendImageGet && (
              <ImageAnalysisSlider Images={RecommendImage} />
            )}
          </div>
        </div>
      )}
      
    </main>
  );
};

export default AnalysisImageUpLoader;
