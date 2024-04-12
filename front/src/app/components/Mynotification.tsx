"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';

const MyNotification = () => {
  const { data: session, status } = useSession();
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        if (status === 'authenticated') {
          const response = await fetch(`${process.env.Localhost}/notification`, {
            method: 'GET',
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              'Content-Type': 'application/json',
            },
          });
          if (response.ok) {
            const data = await response.json();
            setNotifications(data);
          } else {
            console.error('Failed to fetch notifications');
          }
        }
      } catch (error) {
        console.error('Error fetching notifications:', error);
      }
    };

    fetchNotifications();
  }, [session, status]);

  const router = useRouter();

  const handleNotificationClick = async (notificationId: number, PostId: number, Category: string) => {
    try {
      // 클릭한 알림의 상태를 변경하여 서버에 요청
      const response = await fetch(`${process.env.Localhost}/notification/${notificationId}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ checked: true }), // checked 값을 true로 설정하여 알림 확인 상태로 변경
      });

      if (response.ok) {
        // 알림 상태가 성공적으로 변경되면 해당 알림과 관련된 글로 이동
        await router.push(`/Community/${SwitchCategory(Category)}/${PostId}`);
      } else {
        console.error('Failed to update notification status');
      }

    } catch (error) {
      console.error('Error handling notification click:', error);
    }
  };

  const SwitchCategory = (category: any) => {
    switch (category) {
      case "자유":
        return "FreePost";
      case "인기":
        return "PopularityPost";
      case "삽니다":
        return "BuyPost";
      case "팝니다":
        return "SellPost";
      default:
        return "";
    }
  };

  return (
    <>
      <div className="absolute left-[647px] top-[149px] container mx-auto p-4">
        <ul>
          {notifications.map((notification) => (
            <li
              key={notification.notification_id}
              className={`border-b py-2 w-[450px] ${notification.checked ? 'bg-gray-200' : ''}`}
              onClick={() => handleNotificationClick(notification.notification_id, notification.post_id, notification.category)} // 알림 클릭 이벤트 추가
            >
              <p className="text-lg">{notification.content}</p>
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

export default MyNotification;
