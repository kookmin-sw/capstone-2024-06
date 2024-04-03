import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { useSession } from 'next-auth/react';

const MyNotification = () => {
  const { data: session, status } = useSession();
  const [notifications, setNotifications] = useState([]);
  const router = useRouter();

  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        if (status === 'authenticated') {
          const response = await fetch(`${process.env.Localhost}/user/modification`, {
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

  const handleNotificationClick = (notificationId) => {
    router.push(`/notifications/${notificationId}`);
  };

  return (
    <>
      <div className="absolute left-[647px] top-[149px] container mx-auto p-4">
        <ul>
          {notifications.map((notification) => (
            <li key={notification.id} className="border-b py-2 w-[450px]" onClick={() => handleNotificationClick(notification.id)}>
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

export default MyNotification;
