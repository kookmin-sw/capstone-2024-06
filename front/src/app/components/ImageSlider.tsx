"use client";
import React, { useState, useRef } from "react";
import Image from "next/image";

import { Swiper, SwiperSlide } from "swiper/react";
import { Pagination, Navigation } from "swiper/modules";
import { EffectCreative } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";

const ImageSlider = () => {
  const DeskImages = ["/desk4.jpg", "/desk5.png", "/desk6.jpg"];

  return (
    <main className="w-full">
      <div className="swiper-container w-[500px] h-[500px]">
        <Swiper
          slidesPerView={1}
          loop={true}
          pagination={{
            clickable: true,
          }}
          grabCursor={true}
          effect={"creative"}
          creativeEffect={{
            prev: {
              shadow: true,
              translate: [0, 0, -400],
            },
            next: {
              translate: ["100%", 0, 0],
            },
          }}
          modules={[Pagination, EffectCreative]}
          className="mySwiper w-full h-full"
        >
          {DeskImages.map((src, index) => (
            <SwiperSlide key={index}>
              
              <Image
                src={src}
                alt={`Desk ${index + 1}`}
                layout="fill"
                objectFit="cover"
                className="cursor-pointer transition-transform hover:scale-105"
              />
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
    </main>
  );
};

export default ImageSlider;
