"use client";
import Image from "next/image";
import React, { useState } from "react";

const MyProfile = () => {
  const ProfileDummyData = {
    User: "어떤데스크 유저",
    ProfileImage: "/Profilex.png",
  };

  const [Liked, setLiked] = useState(false);
  const LikedBtClick = () => {
    setLiked(!Liked);
  };

  return (
    <main className="flex border w-full h-[30px] justify-center items-center mt-1">
      <Image
        src={ProfileDummyData.ProfileImage}
        alt="name"
        width={30}
        height={30}
        className="cursor-pointer mr-2"
      />

      <div className="w-full font-bold text-sm">{ProfileDummyData.User}</div>
      
    </main>
  );
};

export default MyProfile;
