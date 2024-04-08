"use client";
import { useState } from "react";
import { useSession } from "next-auth/react";
import Nav from "../../components/Nav";
import UserProfile from "../../components/UserProfile";
import UserPosts from "../../components/UserPost";

const Notification = () => {

  return (
    <>
      <main className="flex-col justify-center w-full h-full">
        <Nav />
        <div className="flex justify-center w-full h-auto">
          <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
            <UserProfile />
            <UserPosts />
          </div>
        </div>
      </main>
    </>
  );
};

export default Notification;