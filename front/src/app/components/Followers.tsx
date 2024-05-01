"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from "next/image";


const Follower = () => {
  const { data: session } = useSession();
  const router = useRouter();

  const [followerlist, setFollowerlist] = useState([]);

  useEffect(() => {
    const fetchFollowings = async () => {
      try {

        // 팔로잉 정보를 가져오는 API 요청
        const res = await fetch(`${process.env.Localhost}/user/follower/${session?.user?.user_id}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${(session as any)?.access_token}`,
          },
        });

        // 요청 성공 시 데이터 설정
        if (res.ok) {
          const data = await res.json();
          setFollowerlist(data);
        } else {
          console.error('Failed to fetch followings');
        }
      } catch (error) {
        console.error(error);
      }
    };

    fetchFollowings();
  }, [session]);

  const handleAuthorImageClick = (user_id: string) => {
    router.push(`/Users?user_id=${user_id}`);
    // Users 페이지로 이동
  };


  return (
    <div>
      <div className="absolute w-[173px] h-[83px] left-[697px] top-[99px] font-semibold text-base leading-9 text-black hover:text-[#F4A460]"
      >
        팔로워
      </div>
      {followerlist && followerlist.map((follower) => (
        <div key={follower.user_id}>
          <h2>{follower.name}</h2>
          <Image src={follower.image} width={40} height={30} alt={""} className="rounded-full"
            onClick={() => handleAuthorImageClick(follower.image)}
          />
          <h2>{follower.email}</h2>
        </div>
      ))}
    </div>
  );
};

export default Follower;