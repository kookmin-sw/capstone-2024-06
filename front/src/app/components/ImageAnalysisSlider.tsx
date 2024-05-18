"use client";
import React, { useState } from "react";
import { useSession } from "next-auth/react";
import Image from "next/image";
import Modal from "./Modal";

import { Swiper, SwiperSlide } from "swiper/react";
import { FreeMode, Navigation } from "swiper/modules";

import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";
import ViewDetail from "./ViewDetail";

interface ImageItem {
  index: number;
  src_url: string;
  landing: string;
}

const ImageAnalysisSlider = ({ Images }: { Images: ImageItem[] }) => {
  const { data: session } = useSession();

  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [ImageLanding, SetImageLanding] = useState("");

  const handleImageClick = (imageUrl: string, imageLanding: string) => {
    setPreviewImage(imageUrl);
    SetImageLanding(imageLanding);
  };

  const closePreview = () => {
    setPreviewImage(null);
  };

  const [ViewDetails, SetViewDetails] = useState<
    {
      color: string;
      items: { name: string; landing: string; src_url: string };
    }[]
  >([]);

  const [ViewDetailOn, SetViewDetailOn] = useState(false);
  const [IndexCheck, SetIndexCheck] = useState(0);

  const ViewDetailBtClick = (index: number) => {
    if (index === IndexCheck && ViewDetailOn) {
      SetViewDetailOn(false);
      return;
    } else if (index === IndexCheck && !ViewDetailOn) {
      SetViewDetailOn(true);
      console.log("set ture");
      return;
    } else {
      SetIndexCheck(index);
    }
    const ItemGet = async () => {
      try {
        const ImagePost = await fetch(
          `${process.env.Localhost}/recommend/item?index=${index}`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
          }
        );
        const ItemDatas = await ImagePost.json();
        SetViewDetails(ItemDatas);
        if (ViewDetailOn) {
          SetViewDetailOn(false);
          SetViewDetailOn(true);
        } else {
          SetViewDetailOn(true);
        }
      } catch (error) {
        console.error("Error", error);
      }
    };
    ItemGet();
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
          {Images.map((src, index) => (
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
                  className="cursor-pointer transition-transform hover:scale-105"
                />
              </div>
              <div className="w-full flex items-center justify-center mt-3">
                <button
                  onClick={() => ViewDetailBtClick(src.index)}
                  className="font-semibold border w-12 h-12 rounded cursor-pointer"
                > 
                  <div className="flex justify-center items-center">
                    <img
                      src={`/+.png`}
                      alt={`Desk ${index + 1}`}
                      className=" w-6 h-6 transition-transform hover:scale-105"
                    />
                  </div>
                </button>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
      {ViewDetailOn && <ViewDetail Items={ViewDetails as any} />}
      {previewImage && (
        <Modal
          imageUrl={previewImage}
          onClose={closePreview}
          imageLanding={ImageLanding}
        />
      )}
    </main>
  );
};

export default ImageAnalysisSlider;
