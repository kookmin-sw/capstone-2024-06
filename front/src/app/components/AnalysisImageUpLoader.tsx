"use client";
import { useState, ChangeEvent, DragEvent, useEffect } from "react";
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
  const [plotlyHTML, setPlotlyHTML] = useState("");
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
  const [SampleImage, SetSampleImage] = useState([
    {
      index: 0,
      src_url: "",
      landing: "",
      score: 0,
    },
  ]);
  useEffect(() => {
    const SampleImageGet = async () => {
      try {
        const ImagePost = await fetch(
          `${process.env.Localhost}/recommend/sample`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
          }
        );
        const SampleImageDatas = await ImagePost.json();
        console.log(SampleImageDatas);
        SetSampleImage(SampleImageDatas);
      } catch (error) {
        console.error("Error", error);
      }
    };
    SampleImageGet();
  }, []);
  const SampleImageScore = (index: number, score: number) => {
    SampleImage[index].score = score;
  };

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
        `${process.env.Localhost}/recommend/preference`,
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
    } catch (error) {
      console.error("Error", error);
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
                  분석중
                  <FontAwesomeIcon
                    icon={faSpinner}
                    className="animate-spin ml-1"
                  />
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
      {/* {!AnalyBtClick && <RecommendImgSlider Images={images} />}
      <iframe
        className="plot"
        srcDoc={plotlyHTML}
        width="1000"
        height="800"
      ></iframe> */}
      <div className="flex mt-10">
        {SampleImage.map((sample, index) => (
          <div key={sample.index} className="flex-col border w-1/5 h-[200px]">
            <div
              style={{ width: "100%", height: "100%", position: "relative" }}
            >
              <Image
                src={sample.src_url}
                alt="Sample Image"
                layout="fill"
                objectFit="cover"
              />
            </div>
            <div className="mx-5 flex flex-row-reverse justify-center text-2xl">
              <input
                type="radio"
                className="peer hidden"
                id={`value5_${index}`}
                value="5"
                name={`score_${index}`}
                onChange={() => SampleImageScore(index, 5)}
              />
              <label
                htmlFor={`value5_${index}`}
                className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-600"
              >
                ★
              </label>
              <input
                type="radio"
                className="peer hidden"
                id={`value4_${index}`}
                value="4"
                name={`score_${index}`}
                onChange={() => SampleImageScore(index, 4)}
              />
              <label
                htmlFor={`value4_${index}`}
                className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-600"
              >
                ★
              </label>
              <input
                type="radio"
                className="peer hidden"
                id={`value3_${index}`}
                value="3"
                name={`score_${index}`}
                onChange={() => SampleImageScore(index, 3)}
              />
              <label
                htmlFor={`value3_${index}`}
                className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-600"
              >
                ★
              </label>
              <input
                type="radio"
                className="peer hidden"
                id={`value2_${index}`}
                value="2"
                name={`score_${index}`}
                onChange={() => SampleImageScore(index, 2)}
              />
              <label
                htmlFor={`value2_${index}`}
                className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-600"
              >
                ★
              </label>
              <input
                type="radio"
                className="peer hidden"
                id={`value1_${index}`}
                value="1"
                name={`score_${index}`}
                onChange={() => SampleImageScore(index, 1)}
              />
              <label
                htmlFor={`value1_${index}`}
                className="cursor-pointer peer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-600"
              >
                ★
              </label>
            </div>
          </div>
        ))}
      </div>
      <div className="flex mt-10">
        {RecommendImage.map((sample, index) => (
          <div key={sample.index} className="flex-col border w-1/5 h-[200px]">
            <div
              style={{ width: "100%", height: "100%", position: "relative" }}
            >
              <Image
                src={sample.src_url}
                alt="Sample Image"
                layout="fill"
                objectFit="cover"
              />
            </div>
          </div>
        ))}
      </div>
      <button className="border" onClick={SampleImageScoreSend}>
        Test
      </button>
    </main>
  );
};

export default AnalysisImageUpLoader;
