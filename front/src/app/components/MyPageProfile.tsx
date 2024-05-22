"use client";
import Image from "next/image";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { Session } from "next-auth";

interface ExtendedSession extends Session {
  user?: {
    name?: string | null;
    email?: string | null;
    image?: string | null;
    user_id: string;
  }
}

const MyPageProfile = () => {
  const { data: session } = useSession();
  const router = useRouter();

  const [MyPageProfile, setMyPageProfile] = useState<{
    name: string;
    email: string;
    image: string;
    user_id: string;
    followed: boolean;
    follower_count: number;
    followee_count: number;
  }>();

  useEffect(() => {
    if (!session) {
      return;
    }
    const fetchUserData = async () => {
      try {
        if (!session) return;

        const response = await fetch(`${process.env.Localhost}/user/profile/${(session as ExtendedSession)?.user?.user_id}`, {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        });
        if (response.ok) {
          const userData = await response.json();
          console.log(userData);
          setMyPageProfile(userData);
        } else {
          console.error('Failed to fetch user data');
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
      }
    };

    fetchUserData();
  }, [session]);


  const [MyPageProfileIcons, SetMyPageProfileIcons] = useState([
    "/note.png",
    "/chart.png",
    "/bookmarks.png",
  ]);

  const MyPageProfileName = ["내 글", "팔로우", "스크랩", "메세지"]

  const handleClick = async (index : any) => {
    switch (index) {
      case 0:
        router.push("/Mypage/MyPost");
        break;
      case 1:
        router.push("/Mypage/MyAnalysis");
        break;
      case 2:
        router.push("/Mypage/Scrap");
        break;
      default:
        break;
    }
  };

  const EditProfileClick = () => {
    router.push("/Mypage/EditProfile")
  }

  const FollowingClick = () => {
    router.push("/Mypage/Following")
  }

  const FollowerClick = () => {
    router.push("/Mypage/Follower")
  }


  return (
    <main className="w-[600px] mt-10 border h-[400px] rounded-md">
      <div className="flex-col">
        <div className="flex w-full h-[150px]">
          <div className="flex h-full w-1/2 justify-center items-center ">
            <div className="w-[100px] h-[100px] flex justify-center items-center">
              <Image
                src={session?.user?.image ?? ""}
                alt="Profile image"
                width={1000}
                height={1000}
                objectFit="cover"
                className="cursor-pointer mr-1 rounded-full"
              />
            </div>
          </div>
          <div className="flex w-1/2 h-full items-center flex justify-center items-center">
            <div className="text-2xl">{session?.user?.name?.substring(0, 4)}</div>
          </div>
        </div>
        <div className="flex justify-center items-center w-full h-[60px]">
          <div className="w-[250px] bg-[#ced4da] font-bold rounded-sm h-[30px] flex items-center justify-center hover:bg-[#F4A460]"
            onClick={EditProfileClick}>
            프로필 수정
          </div>
        </div>
        <div className="flex justify-center h-[60px]">
          <div className="mr-1 hover:text-[#F4A460]"
            onClick={FollowingClick}>팔로잉 {MyPageProfile?.followee_count}</div>
          <div className="mr-1 hover:text-[#F4A460]"
            onClick={FollowerClick}>팔로워 {MyPageProfile?.follower_count}</div>
        </div>
        <div className="flex w-full h-[100px] justify-center items-center space-x-5">
          {MyPageProfileIcons.map((icon, index) => (
            <div key={index}>
              <div className="w-[40px] h-[40px] mb-2">
                <Image
                  src={icon}
                  alt="Profile image"
                  width={100}
                  height={1000}
                  objectFit="cover"
                  className="cursor-pointer mr-1"
                  onClick={() => handleClick(index)}
                />
              </div>
              <div>{MyPageProfileName[index]}</div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
};

export default MyPageProfile;