"use client";
import Image from "next/image";
import React from "react";

interface MyProfileProps {
  UserName: string;
  UserProfile: string;
}

const MyProfile: React.FC<MyProfileProps> = ({ UserName, UserProfile }) => {
  if (UserProfile === null) {
    UserProfile = "/Profilex2.webp";
  }

  const ProfileDummyData = {
    User: UserName,
    ProfileImage: UserProfile,
  };

  return (
    <main className="flex w-1/3 h-full items-center ml-1">
      <div className="flex w-[20px] h-[20px] mr-1">
        <Image
          src={ProfileDummyData.ProfileImage}
          alt="name"
          width={100}
          height={100}
          className="cursor-pointer rounded-full"
        />
      </div>
      <div className="flex items-center font-semibold text-[10.5px]">{ProfileDummyData.User.length > 10 ? `${ProfileDummyData.User.slice(0, 5)}...` : ProfileDummyData.User}</div>
    </main>
  );
};

export default MyProfile;
