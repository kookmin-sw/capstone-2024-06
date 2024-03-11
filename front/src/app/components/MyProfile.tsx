"use client";
import Image from "next/image";
import React from "react";

interface MyProfileProps {
  UserName: string;
  UserProfile: string;
}

const MyProfile: React.FC<MyProfileProps> = ({UserName, UserProfile}) => {

  const ProfileDummyData = {
    User: UserName,
    ProfileImage: UserProfile,
  };


  return (
    <main className="flex mb-2 w-full h-[30px] justify-center items-center mt-1">
      <Image
        src={ProfileDummyData.ProfileImage}
        alt="name"
        width={30}
        height={20}
        className="cursor-pointer mr-1"
      />
      <div className="font-semibold text-l">{ProfileDummyData.User}</div>
      
    </main>
  );
};

export default MyProfile;
