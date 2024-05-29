"use client";
import React from "react";
import ItemRecommend from "./ItemRecommend";

type RecommendItem = {
  name: string;
  landing: string;
  src_url: string;
};

const RecommendViewDetail = ({
  RecommendItems,
}: {
  RecommendItems: RecommendItem[];
}) => {
  console.log(RecommendItems);
  return (
    <main>
      <div className="mt-10 w-full h-fit border flex-col p-4 rounded">
        <div className="flex w-fit h-fit items-center justify-center">
          <div className="my-2 text-xl mr-2 font-semibold">맞춤 추천</div>
        </div>
        <div className="flex w-full border">
          <ItemRecommend Ritems={RecommendItems} />
        </div>
      </div>
    </main>
  );
};

export default RecommendViewDetail;
