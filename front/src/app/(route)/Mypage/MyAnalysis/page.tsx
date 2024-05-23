"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import Nav from "../../../components/Nav";
import ImageAnalysisSlider from "@/app/components/ImageAnalysisSlider";

const EditPost = () => {
  const { data: session } = useSession();
  const [LastAnalysisData, SetLastAnalysisData] = useState<
    Array<{
      index: number;
      src_url: string;
      landing: string;
    }>
  >([]);
  useEffect(() => {
    if (!session) return;
    const MyAnalysis = async () => {
      try {
        const response = await fetch(
          `/api/recommend/reload`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              "Content-Type": "application/json",
            },
          }
        );
        const data = await response.json();
        console.log(data);
        SetLastAnalysisData(data);
      } catch (error) {
        console.error("Error", error);
      }
    };
    MyAnalysis();
  }, [session]);

  return (
    <main className="flex-col justify-center w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <div className="w-fit h-fit font-inter font-semibold text-4xl text-yellow-600 my-20">
            어떤데스크
          </div>
          <div className="mb-10 text-xl font-semibold text-[#808080]">마지막 분석 결과</div>
          <div className="w-fit">
            {LastAnalysisData.length > 0 && (
              <ImageAnalysisSlider Images={LastAnalysisData} />
            )}
          </div>
        </div>
      </div>
    </main>
  );
};

export default EditPost;
