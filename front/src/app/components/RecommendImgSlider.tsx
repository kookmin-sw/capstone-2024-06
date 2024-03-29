"use client";
import React, { useState } from "react";
import Image from "next/image";
import Modal from "./Modal";

import { Swiper, SwiperSlide } from "swiper/react";
import { Pagination, FreeMode } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";

const RecommendImgSlider = ({ Images }: { Images: string[] }) => {
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
          modules={[FreeMode, Pagination]}
          className="mySwiper"
        >
          {Images.map((src, index) => (
            <SwiperSlide key={index}>
              <div
                className="w-[300px] h-[300px] relative"
                onClick={() =>
                  handleImageClick(`${process.env.Localhost}${src}`)
                }
              >
                <Image
                  src={`${process.env.Localhost}${src}`}
                  alt={`Desk ${index + 1}`}
                  layout="fill"
                  objectFit="cover"
                  className="cursor-pointer transition-transform transform hover:scale-105"
                />
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
      {previewImage && <Modal imageUrl={previewImage} onClose={closePreview} />}
    </main>
  );
};

export default RecommendImgSlider;
