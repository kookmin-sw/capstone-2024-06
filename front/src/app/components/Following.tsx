"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from "next/image";
import { Session } from 'next-auth';

interface ExtendedSession extends Session {
  user?: {
    name?: string | null;
    email?: string | null;
    image?: string | null;
    user_id: string;
  }
}

const Following = () => {
  const { data: session } = useSession();
  const router = useRouter();

  const [followinglist, setFollowinglist] = useState<Array<{
    "name": string;
    "email": string;
    "image": string;
    "user_id": string;
    "followed": boolean
  }>>([]);

  useEffect(() => {
    const fetchFollowings = async () => {
      try {
        if (!session) return;

        // 팔로잉 정보를 가져오는 API 요청
        const res = await fetch(`/what-desk-api/user/followee/${(session as ExtendedSession)?.user?.user_id}`, {
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

  const HomeImageClick = () => {
    router.push("/Mypage")
  }

  return (
    <div>
      <div className="absolute w-[173px] h-[83px] left-[697px] top-[99px] font-semibold text-base leading-9 text-black hover:text-[#F4A460]"
      >
        팔로잉
      </div>
      <div className="absolute w-[173px] h-[83px] left-[1099px] top-[119px]">
        <Image
          src="/house.png"
          alt="home Image"
          width={100}
          height={100}
          style={{ width: "auto", height: "auto" }}
          onClick={HomeImageClick}
        />
      </div>
      <div className="absolute left-[670px] top-[180px]">
        {followinglist.map((following) => (
          <div key={following.user_id} className="flex items-center space-x-2">
            <Image
              src={following.image}
              width={40}
              height={30}
              alt={""}
              className="rounded-full"
              onClick={() => handleAuthorImageClick(following.user_id)}
            />
            <div>
              <h2>{following.name + " " + following.email}</h2>
            </div>
          </div>
        ))}
      </div>


    </div>
  );
};

export default Following;