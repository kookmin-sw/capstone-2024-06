"use client";
import React, {
  useState,
  ChangeEvent,
  DragEvent,
  FunctionComponent,
} from "react";

import axios from "axios";

interface ImagePreview {
  url: string;
  file: File;
}

const PictureUpload: FunctionComponent = () => {
  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null);  


  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
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

  const ImageAnalysis = async () => {
    try {
      if (imagePreview) {
        const formData = new FormData();
        formData.append("image", imagePreview.file);

        const response = await axios.post(
          "http://210.178.142.51:8002/process_image/",
          formData
        );
        console.log(response);
      }
    } catch (error) {
      console.error(error);
    }
  };
  
  return (
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
          <div className="flex items-center justify-center border-dashed border-2 text-[#808080] text-sm w-[800px] h-[400px] cursor-pointer">
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
            onChange={handleImageChange}
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
          >
            파일 선택
          </label>
        </div>
      ) : (
        <div className="flex space-x-2">
          <div
            className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
            onClick={ImageDeleteBt}
          >
            사진제거
          </div>
          <div
            className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
            onClick={ImageAnalysis}
          >
            분석하기
          </div>
        </div>
      )}
    </div>
  );
};

export default PictureUpload;
