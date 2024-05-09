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
  score: number;
}

const RecommendImgSlider = ({ Images }: { Images: ImageItem[] }) => {

  console.log(Images)

  const SampleImageScore = (index: number, score: number) => {
    Images[index].score = score;
  };

  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const handleImageClick = (imageUrl: string) => {
    setPreviewImage(imageUrl);
  };

  const closePreview = () => {
    setPreviewImage(null);
  };

  return (
    <main className="flex w-full justify-center ">
      <div className="swiper-container w-[1000px] h-fit flex">
        <Swiper
          slidesPerView={3}
          spaceBetween={30}
          freeMode={true}
          navigation = {true}
          modules={[FreeMode]}
          className="mySwiper"
        >
          {Images.map((src, index) => (
            <SwiperSlide key={index}>
              <div
                className="w-[300px] h-[300px] relative"
                onClick={() =>
                  handleImageClick(`${src.src_url}`)
                }
              >
                <Image
                  src={`${src.src_url}&w=300&h=300&c=c&q=80`}
                  alt={`Desk ${index + 1}`}
                  layout="fill"
                  objectFit="cover"
                  className="cursor-pointer transition-transform transform hover:scale-105"
                />
                
              </div>
              <div className="mx-5 flex flex-row-reverse justify-center text-2xl">
                  <input
                    type="radio"
                    className="peer hidden"
                    id={`value5_${index}`}
                    value="5"
                    name={`score_${index}`}
                    onChange={() => SampleImageScore(index, 5)}
                  />
                  <label
                    htmlFor={`value5_${index}`}
                    className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-400"
                  >
                    ★
                  </label>
                  <input
                    type="radio"
                    className="peer hidden"
                    id={`value4_${index}`}
                    value="4"
                    name={`score_${index}`}
                    onChange={() => SampleImageScore(index, 4)}
                  />
                  <label
                    htmlFor={`value4_${index}`}
                    className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-400"
                  >
                    ★
                  </label>
                  <input
                    type="radio"
                    className="peer hidden"
                    id={`value3_${index}`}
                    value="3"
                    name={`score_${index}`}
                    onChange={() => SampleImageScore(index, 3)}
                  />
                  <label
                    htmlFor={`value3_${index}`}
                    className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-400"
                  >
                    ★
                  </label>
                  <input
                    type="radio"
                    className="peer hidden"
                    id={`value2_${index}`}
                    value="2"
                    name={`score_${index}`}
                    onChange={() => SampleImageScore(index, 2)}
                  />
                  <label
                    htmlFor={`value2_${index}`}
                    className="cursor-pointer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-400"
                  >
                    ★
                  </label>
                  <input
                    type="radio"
                    className="peer hidden"
                    id={`value1_${index}`}
                    value="1"
                    name={`score_${index}`}
                    onChange={() => SampleImageScore(index, 1)}
                  />
                  <label
                    htmlFor={`value1_${index}`}
                    className="cursor-pointer peer text-gray-400 peer-hover:text-yellow-400 peer-checked:text-yellow-400"
                  >
                    ★
                  </label>
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
