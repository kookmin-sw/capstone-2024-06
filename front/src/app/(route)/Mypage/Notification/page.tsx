"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Nav from "../../../components/Nav";
import MyPageProfile from "../../../components/MyPageProfile";

const Notification = () => {
  const { data: session } = useSession();
  const router = useRouter();
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      message: "새로운 알림이 도착했습니다.",
      date: new Date(),
    },
    {
      id: 2,
      message: "누군가가 당신의 게시물에 댓글을 남겼습니다.",
      date: new Date(),
    },
    // 필요에 따라 추가할 수 있습니다.
  ]);

  return (
    <>
      <Nav />
      <MyPageProfile />
      <div className="absolute left-[647px] top-[149px] container mx-auto p-4">
        <ul>
          {notifications.map((notification) => (
            <li key={notification.id} className="border-b py-2 w-[480px]">
              <p className="text-lg">{notification.message}</p>
              <p className="max-w-[400px] text-sm text-gray-500">
                {notification.date.toLocaleString()}
              </p>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
};

export default Notification;
