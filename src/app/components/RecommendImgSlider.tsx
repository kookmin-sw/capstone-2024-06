"use client";
import React, { useState, useRef } from "react";
import Image from "next/image";

import { Swiper, SwiperSlide } from "swiper/react";
import { Pagination} from "swiper/modules";


import "swiper/css";
import "swiper/css/pagination";


const RecommendImgSlider = () => {
  const DeskImages = [ "/desk1.png", "/desk2.png", "/desk3.png", "/desk4.jpg", "/desk5.png", "/desk6.jpg", "/desk5.png",  "/desk5.png", "/desk5.png", "/desk5.png",];

  return (
    <main>
      <div className="swiper-container w-full h-[500px]">
      <Swiper
        slidesPerView={5}
        spaceBetween={30}
        pagination={{
          clickable: true,
        }}
        modules={[Pagination]}
        className="mySwiper"
      >
          {DeskImages.map((src, index) => (
            <SwiperSlide key={index}>
              
              <Image
                src={src}
                alt={`Desk ${index + 1}`}
                objectFit="cover"
                className="cursor-pointer transition-transform hover:scale-105"
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
