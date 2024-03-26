"use client";
import React from "react";
import Image from "next/image";

import { Swiper, SwiperSlide } from "swiper/react";
import { Pagination, FreeMode } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";

const RecommendImgSlider = ({ Images }: { Images: string[] }) => {
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
              <Image
              src={src}
                // src={`${process.env.Localhost}${src}`}
                alt={`Desk ${index + 1}`}
                width={300}
                height={300}
              
              />
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
    </main>
  );
};

export default RecommendImgSlider;
