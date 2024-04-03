"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { useSession } from 'next-auth/react';

const Following = () => {
  const { data: session } = useSession();
  const [followings, setFollowings] = useState([]);

  useEffect(() => {
    const fetchFollowings = async () => {
      try {
        // // 세션이 없으면 홈페이지로 리다이렉션
        // if (!session) {
        //   router.push('/');
        //   return;
        // }

        // 팔로잉 정보를 가져오는 API 엔드포인트로 요청
        const res = await fetch(`${process.env.LOCALHOST}/api/followings/${session?.user?.user_id}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${(session as any)?.access_token}`,
          },
        });

        // 요청 성공 시 데이터 설정
        if (res.ok) {
          const data = await res.json();
          setFollowings(data);
        } else {
          console.error('Failed to fetch followings');
        }
      } catch (error) {
        console.error(error);
      }
    };

    fetchFollowings();
  }, [session]);



  return (
    <div>
      <h1>Following List</h1>
      {/* 팔로잉 목록을 매핑하여 출력 */}
      {followings.map((following) => (
        <div key={following.id}>
          <h2>{following.username}</h2>
          <img src={following.avatar} alt={following.username} />
        </div>
      ))}
    </div>
  );
};

export default Following;
