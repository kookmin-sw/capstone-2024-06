import Image from "next/image";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";

const UserProfile = () => {
  const { data: session } = useSession();

  const [userProfile, setUserProfile] = useState(null);
  const [followerCount, setFollowerCount] = useState(0);
  const [followingCount, setFollowingCount] = useState(0);
  const [isFollowing, setIsFollowing] = useState(false);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await fetch(`${process.env.Localhost}/user/${userId}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (response.ok) {
          const userData = await response.json();
          setUserProfile(userData);
          setFollowerCount(userData.followerCount);
          setFollowingCount(userData.followingCount);
          setIsFollowing(userData.isFollowing);
        } else {
          console.error('Failed to fetch user data');
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
      }
    };

    fetchUserData();
  }, []);


  // 팔로우 또는 언팔로우 요청을 보내는 함수
  const handleFollowToggle = async () => {
    try {
      const response = await fetch(`${process.env.Localhost}/user/follow`, {
        method: isFollowing ? 'DELETE' : 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userId: userProfile.id }), // 사용자 ID를 전송
      });
      if (response.ok) {
        setIsFollowing(!isFollowing); // 팔로우 상태 업데이트
      } else {
        console.error('Failed to toggle follow status');
      }
    } catch (error) {
      console.error('Error toggling follow status:', error);
    }
  }


  return (
    <div className="flex flex-col items-center w-full">
      <div className="flex flex-col items-center w-full">
        <div className="flex flex-col items-center w-11/12">
          <div className="flex justify-between w-full mt-6">
            <div className="flex items-center">
              {userProfile && (
                <Image
                  src={userProfile.image}
                  alt={userProfile.name}
                  width={80}
                  height={80}
                />
              )}
              <div className="flex flex-col ml-4">
                <div className="flex items-center">
                  <h1 className="text-2xl font-bold">{userProfile ? userProfile.name : "User Name"}</h1>
                  {/* 팔로우 버튼 또는 언팔로우 버튼 표시 */}
                  <button
                    className={`ml-4 bg-[#FFD600] text-black rounded-md px-2 py-1 ${isFollowing ? 'bg-red-500' : ''}`}
                    onClick={handleFollowToggle}
                  >
                    {isFollowing ? '언팔로우' : '팔로우'}
                  </button>
                </div>
                <div className="flex items-center mt-2">
                  <h1 className="text-sm hover:text-[#F4A460]">팔로워 {followerCount}</h1>
                </div>
                <div className="flex items-center mt-2">
                  <h1 className="text-sm hover:text-[#F4A460]">팔로잉 {followingCount}</h1>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default UserProfile;
