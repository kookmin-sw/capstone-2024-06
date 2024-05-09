"use client";
import React, { useState } from "react";
import Image from "next/image";
import Modal from "./Modal";

import { Swiper, SwiperSlide } from "swiper/react";
import { FreeMode, Navigation } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";

interface ImageItem {
  index: number;
  src_url: string;
  landing: string;
}

const ImageAnalysisSlider = ({ Images }: { Images: ImageItem[] }) => {
  console.log(Images);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const handleImageClick = (imageUrl: string) => {
    setPreviewImage(imageUrl);
  };

  const closePreview = () => {
    setPreviewImage(null);
  };

  return (
    <main className="flex w-full justify-center ">
      <div className="swiper-container w-[1000px] h-fit">
        <Swiper
          slidesPerView={3}
          spaceBetween={30}
          freeMode={true}
          pagination={{
            clickable: true,
          }}
          navigation={true}
          modules={[FreeMode, Navigation]}
          className="mySwiper"
        >
          {Images.map((src, index) => (
            <SwiperSlide key={index}>
              <div
                className="w-[300px] h-[300px] relative"
                onClick={() => handleImageClick(`${src.src_url}`)}
              >
                <Image
                  src={`${src.src_url}&w=300&h=300&c=c&q=80`}
                  alt={`Desk ${index + 1}`}
                  layout="fill"
                  objectFit="cover"
                  className="cursor-pointer transition-transform transform hover:scale-105"
                />
              </div>
              <div className="w-full flex items-center justify-end pr-3">
                <button className="text-sm">더보기</button>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
      {previewImage && <Modal imageUrl={previewImage} onClose={closePreview} />}
    </main>
  );
};

export default ImageAnalysisSlider;
