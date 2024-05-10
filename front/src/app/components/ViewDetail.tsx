"use client";
import React from "react";
import Image from "next/image";
import ItemRecommend from "./ItemRecommend";

const ViewDetail = ({
  Items,
}: {
  Items: {
    color: string;
    items: { name: string; landing: string; src_url: string }[];
  }[];
}) => {
  console.log(Items);
  return (
    <main>
      {Items.map((src, index) => (
        <div key={index} className="mt-10 w-full h-fit border flex-col p-4 rounded">
          <div className="flex w-fit h-fit items-center justify-center">
            <div className="my-2 text-xl mr-2 font-semibold">추천 색상</div>
            <div
              className="border w-6 h-6 rounded-full"
              style={{ backgroundColor: src.color }}
            ></div>
          </div>
          <div className="flex w-full border">
            <ItemRecommend Ritems={src.items} />
          </div>
        </div>
      ))}
    </main>
  );
};

export default ViewDetail;
