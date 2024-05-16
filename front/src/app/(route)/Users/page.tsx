"use client";
import { useState } from "react";
import { useSession } from "next-auth/react";
import Nav from "../../components/Nav";
import User from "../../components/UserPage";

const Notification = () => {

  return (
    <>
      <main className="flex-col justify-center w-full h-full">
        <Nav />
        <div className="flex justify-center w-full h-auto">
          <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
            <User />
          </div>
        </div>
      </main>
    </>
  );
};

export default Notification;