"use client";
import React, { useState, useRef } from "react";
import Image from "next/image";

import { Swiper, SwiperSlide } from "swiper/react";
import { Pagination, Autoplay, Navigation} from "swiper/modules";


import "swiper/css";
import "swiper/css/pagination";
import 'swiper/css/navigation';



const RecommendImgSlider = () => {
  const DeskImages = [ "/desk1.png", "/desk2.png", "/desk3.png", "/desk4.jpg", "/desk5.png", "/desk6.jpg", "/desk5.png",  "/desk5.png", "/desk5.png", "/desk5.png",];

  return (
    <main>
      <div className="swiper-container w-full h-[500px]">
      <Swiper
        spaceBetween={30}
        centeredSlides={true}
        autoplay={{
          delay: 2500,
          disableOnInteraction: false,
        }}
        pagination={{
          clickable: true,
        }}
        navigation={true}
        modules={[Autoplay, Pagination, Navigation]}
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
