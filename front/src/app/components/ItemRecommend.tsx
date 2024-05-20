"use client";
import React, { useState } from "react";
import Image from "next/image";
import Modal from "./Modal";

import { Swiper, SwiperSlide } from "swiper/react";
import { FreeMode, Navigation } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";

interface RecommendItem {
  name: string;
  src_url: string;
  landing: string;
}

const ItemRecommend = ({ Ritems}: { Ritems: RecommendItem[] }) => {

  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [ImageLanding, SetImageLanding] = useState("")

  const handleImageClick = (imageUrl: string, imageLanding : string) => {
    setPreviewImage(imageUrl);
    SetImageLanding(imageLanding)
  };

  const closePreview = () => {
    setPreviewImage(null);
  };

  return (
    <main className="flex-col w-full justify-center ">
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
          {Ritems.map((src, index) => (
            <SwiperSlide key={index}>
              <div
                className="w-[300px] h-[300px] relative"
                onClick={() => handleImageClick(src.src_url, src.landing)}
              >
                <Image
                  src={`${src.src_url}&w=300&h=300&c=c&q=80`}
                  alt={`Desk ${index + 1}`}
                  layout="fill"
                  objectFit="cover"
                  className="cursor-pointer transition-transform transform hover:scale-105"
                />
              </div>
              <div className="w-full mt-1 text-center font-semibold text-sm">
                  {src.name}
                </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
      {previewImage && <Modal imageUrl={previewImage} onClose={closePreview} imageLanding={ImageLanding} />}
    </main>
  );
};

export default ItemRecommend;
