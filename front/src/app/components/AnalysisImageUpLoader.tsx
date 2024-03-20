"use client";
import { useState, ChangeEvent, DragEvent } from "react";
import { useSession } from "next-auth/react";
import Image from "next/image";

interface ImagePreview {
  url: string;
  file: File;
}

const AnalysisImageUpLoader = () => {
  const { data: session } = useSession();
  const [images, setImages] = useState<string[]>([]);

  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null);

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
        const formData = new FormData();
        formData.append("file", imagePreview.file);
        const ImagePost = await fetch(
          `${process.env.Localhost}/process_image`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
            body: formData,
          }
        );
        const ImageDatas = await ImagePost.json();
        console.log(ImageDatas)
        console.log(ImageDatas.file_name)
        setImages(ImageDatas.file_name);
      }
    } catch (error) {
      console.error("Error", error);
    }
  };

  return (
    <main>
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
      {images.length > 0 &&
        images.map((fileName: string, index: number) => (
          <div key={index} className="flex w-full">
            <Image
              src={`${process.env.Localhost}/result/${fileName}`}
              alt="Post Image"
              width={1000}
              height={1000}
              objectFit="cover"
              priority
            />
          </div>
        ))}
    </main>
  );
};

export default AnalysisImageUpLoader;
