"use client";
import React from "react";
import Image from "next/image";

import { Swiper, SwiperSlide } from "swiper/react";
import { Pagination, Navigation } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";

const RecommendImgSlider = ({ Images }: { Images: string[] }) => {
  return (
    <main className="flex w-full justify-center mt-20">
      <div className="swiper-container w-[400px] border h-fit">
        <Swiper
          spaceBetween={30}
          centeredSlides={true}
          pagination={{
            clickable: true,
          }}
          navigation={true}
          modules={[Pagination, Navigation]}
          className="mySwiper"
        >
          {Images.map((src, index) => (
            <SwiperSlide key={index}>
              <Image
                src={`${process.env.Localhost}${src}`}
                alt={`Desk ${index + 1}`}
                width={400}
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
