"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import Nav from "../../../components/Nav";
import ImageAnalysisSlider from "@/app/components/ImageAnalysisSlider";

const EditPost = () => {
  const { data: session } = useSession();
  const [ LastAnalysisData, SetLastAnalysisData ] = useState([
    { index: 0, src_url: "", landing: "" },
  ])
  useEffect(() => {
    if (!session) return;
    const MyAnalysis = async () => {
      try {
        const response = await fetch(
          `${process.env.Localhost}/recommend/reload`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              "Content-Type": "application/json",
            },
          }
        );
        const data = await response.json();
        console.log(data)
        SetLastAnalysisData(data)
      } catch (error) {
        console.error("Error", error);
      }
    };
    MyAnalysis();
  }, [session]);

  return (
    <main>
      <Nav />
      <div className="relative w-screen h-screen bg-[background-color]">
        <div className="absolute w-[251px] h-[91px] left-[200px] top-[66px] font-inter font-semibold text-4xl leading-14 text-yellow-600">
          어떤데스크
        </div>
        {/* <ImageAnalysisSlider Images={LastAnalysisData} /> */}
      </div>
    </main>
  );
};

export default EditPost;
