"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from "next/image";

const Following = () => {
  const { data: session } = useSession();
  const router = useRouter();

  const [followinglist, setFollowinglist] = useState([]);

  useEffect(() => {
    const fetchFollowings = async () => {
      try {
        // 팔로잉 정보를 가져오는 API 요청
        const res = await fetch(`${process.env.Localhost}/user/followee/${session?.user?.user_id}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${(session as any)?.access_token}`,
          },
        });

        // 요청 성공 시 데이터 설정
        if (res.ok) {
          const data = await res.json();
          setFollowinglist(data);
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
        팔로잉
      </div>
      {followinglist.map((following) => (
        <div key={following.user_id}>
          <h2>{following.name}</h2>
          <img src={following.image} alt={following.username} />
          <Image src={following.image} width={40} height={30} alt={""} className="rounded-full"
            onClick={() => handleAuthorImageClick(following.image)}
          />
          <h2>{following.email}</h2>
        </div>
      ))}
    </div>
  );
};

export default Following;