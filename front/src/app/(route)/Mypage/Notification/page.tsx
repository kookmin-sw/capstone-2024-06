"use client";
import { useState } from "react";
import { useSession } from "next-auth/react";
import Nav from "../../../components/Nav";
import MyPageProfile from "../../../components/MyPageProfile";
import MyPagePosts from "../../../components/MyPagePosts";
import MyNotification from "../../../components/Mynotification";

const Notification = () => {

  return (
    <>
      <main className="flex-col justify-center w-full h-full">
        <Nav />
        <div className="flex justify-center w-full h-auto">
          <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
            <MyPageProfile />
            <MyPagePosts />
            <MyNotification />
          </div>
        </div>
      </main>
    </>
  );
};

export default Notification;