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
          const response = await fetch(`${process.env.Localhost}/user/notification`, {
            method: 'GET',
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              'Content-Type': 'application/json',
            },
          });
          if (response.ok) {
            const data = await response.json();
            console.log(data);
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

  const handleNotificationClick = async (notification_id: number, reference_id: number, Category: string) => {
    try {
      // 클릭한 알림의 상태를 변경하여 서버에 요청
      const response = await fetch(`${process.env.Localhost}/user/notification/${notification_id}`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ checked: true }), // checked 값을 true로 설정하여 알림 확인 상태로 변경
      });

      if (response.ok) {
        if (reference_id) {
          //알림 상태가 변경되면 해당 글로 이동
          await router.push(`/Community/${SwitchCategory(Category)}/${reference_id}`);
        } else {
          // reference_id가 없는 경우 다른 경로로 라우팅
          await router.push(`/Mypage/Follower`);
        }
      } else {
        console.error('Failed to update notification status');
      }

    } catch (error) {
      console.error('Error handling notification click:', error);
    }
  };

  const DeleteNotification = async () => {
    try {
      const response = await fetch(`${process.env.Localhost}/user/notification`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        // 알림 삭제 성공 시 알림 목록 다시 불러오기
        fetchNotifications();
      } else {
        console.error('Failed to delete notification');
      }
    } catch (error) {
      console.error('Error deleting notification:', error);
    }
  }

  const SwitchCategory = (Category: any) => {
    console.log(Category);
    switch (Category) {
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

  const NotificationClick = () => {
    router.push("/Mypage/Notification")
  }

  return (
    <>
      <div className="absolute w-[173px] h-[83px] left-[697px] top-[99px] font-semibold text-base leading-9 text-black hover:text-[#F4A460]"
        onClick={NotificationClick}>
        알림
      </div>
      <div className="absolute left-[647px] top-[149px] container mx-auto p-4">
        <ul>
          {notifications.map((notification) => (
            <li
              key={notification.notification_id}
              className={`border-b py-2 w-[450px] ${notification.checked ? 'bg-gray-200' : ''}`}
              onClick={() => handleNotificationClick(notification.notification_id, notification.reference_id, notification.category)} // 알림 클릭 이벤트 추가
            >
              <p className="text-lg">{notification.content}</p>
              <p className="max-w-[400px] text-sm text-gray-500">
                {/* {notification.date.toLocaleString()} */}
              </p>
            </li>
          ))}
        </ul>
      </div>
      <div className="absolute w-[173px] h-[83px] left-[1030px] top-[115px] text-sm text-black hover:text-[#F4A460]">
        <button
          onClick={() => DeleteNotification()}
        >
          알림 모두 삭제
        </button>
      </div>
    </>
  );
};

export default MyNotification;
