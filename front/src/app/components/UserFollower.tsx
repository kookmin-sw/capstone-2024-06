"use client"
import Image from "next/image";
import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from 'next/navigation';
import { useSession } from "next-auth/react";

const UserFollowerlist = ({ }) => {
  const { data: session } = useSession();

  const [userProfile, setUserProfile] = useState(null);
  const [isFollowing, setIsFollowing] = useState(false);
  const [followerlist, setFollowerlist] = useState([]);


  const params = useSearchParams();
  const user_id = params.get('user_id');

  useEffect(() => {
    // if (!session) {
    //   return;
    // }
    const fetchUserData = async () => {
      try {
        const response = await fetch(`${process.env.Localhost}/user/profile/${user_id}`, {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        });
        if (response.ok) {
          const userData = await response.json();
          console.log(userData);
          setUserProfile(userData);
        } else {
          console.error('Failed to fetch user data');
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
      }
    };

    fetchUserData();
  }, [session]);

  //팔로우 또는 언팔로우 요청을 보내는 함수
  const handleFollowToggle = async () => {
    try {
      const response = await fetch(`${process.env.Localhost}/community/follow/${user_id}`, {
        method: userProfile.followed ? 'DELETE' : 'POST',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ followee_user_id: userProfile.user_id }), // 사용자 ID를 전송
      });
      if (response.ok) {
        setIsFollowing(!userProfile.followed); // 팔로우 상태 업데이트
      } else {
        console.error('Failed to toggle follow status');
      }
    } catch (error) {
      console.error('Error toggling follow status:', error);
    }
  }

  useEffect(() => {
    const fetchFollowers = async () => {
      try {
        if (!session) return;

        // 팔로잉 정보를 가져오는 API 요청
        const res = await fetch(`${process.env.Localhost}/user/follower/${user_id}`, {
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

    fetchFollowers();
  }, [session]);

  const handleAuthorImageClick = (user_id: string) => {
    router.push(`/Users?user_id=${user_id}`);
    // Users 페이지로 이동
  };

  const FollowerClick = () => {
    router.push("/Users/Follower")
  }

  const FollowingClick = () => {
    router.push("/Users/Followee")
  }

  const router = useRouter();


  return (
    <main className="flex-col w-full h-auto justify-center items-center mt-5">
      <div className="flex flex-col items-center w-full">
        <div className="flex flex-col items-center w-full">
          <div className="flex flex-col items-center w-11/12">
            <div className="flex justify-between w-full mt-6 border border-brown-500">
              <div className="flex items-center">
                {userProfile && (
                  <Image
                    src={userProfile?.image}
                    alt="name"
                    width={80}
                    height={80}
                  />
                )}
                <div className="flex flex-col ml-4">
                  <div className="flex items-center">
                    <h1 className="text-2xl font-bold">{userProfile?.name}</h1>
                    {/* 팔로우 버튼 또는 언팔로우 버튼 표시 */}
                    {userProfile && (
                      <button
                        className={`ml-4 bg-[#FFD600] text-black rounded-md px-2 py-1 ${userProfile.followed ? 'bg-red-500' : ''}`}
                        onClick={handleFollowToggle}
                      >
                        {userProfile.followed ? '언팔로우' : '팔로우'}
                      </button>
                    )}
                  </div>
                  <div className="flex items-center mt-2">
                    <h1 className="text-sm hover:text-[#F4A460]" onClick={FollowerClick} >팔로워 {userProfile?.follower_count}</h1>
                  </div>
                  <div className="flex items-center mt-2">
                    <h1 className="text-sm hover:text-[#F4A460]" onClick={FollowingClick}>팔로잉 {userProfile?.followee_count}</h1>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="border-t border-gray-500 w-full mt-4"></div>
      <div className="border-t border-transparent w-full mt-4"></div>
      <div className="absolute w-[173px] h-[83px] left-[200px] top-[200px] font-semibold text-base leading-9 text-black hover:text-[#F4A460]"
      >
        팔로워
      </div>
      <div className="absolute left-[220px] top-[250px]">
        {followerlist.map((follower) => (
          <div key={follower.user_id} className="flex items-center space-x-2">
            <Image
              src={follower.image}
              width={40}
              height={30}
              alt={""}
              className="rounded-full"
              onClick={() => handleAuthorImageClick(follower.user_id)}
            />
            <div>
              <h2>{follower.name + " " + follower.email}</h2>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}

export default UserFollowerlist;
