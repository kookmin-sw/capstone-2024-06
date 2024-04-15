import Image from "next/image";
import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from 'next/navigation';
import { useSession } from "next-auth/react";
import MyProfile from "./MyProfile";

const User = ({ }) => {
  const { data: session } = useSession();

  const [userProfile, setUserProfile] = useState(null);
  const [followerCount, setFollowerCount] = useState(0);
  const [followingCount, setFollowingCount] = useState(0);
  const [isFollowing, setIsFollowing] = useState(false);
  const [userPosts, setUserPosts] = useState([]);

  const params = useSearchParams();
  const user_id = params.get('user_id');

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await fetch(`${process.env.Localhost}/user/${user_id}`, {
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
          setUserPosts(userData.userPosts);
        } else {
          console.error('Failed to fetch user data');
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
      }
    };

    fetchUserData();
  }, []);


  //팔로우 또는 언팔로우 요청을 보내는 함수
  const handleFollowToggle = async () => {
    try {
      const response = await fetch(`${process.env.Localhost}/followees`, {
        method: isFollowing ? 'DELETE' : 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ followee_user_id: userProfile.id }), // 사용자 ID를 전송
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

  const router = useRouter();

  const PostClick = (PostId: number, Category: string) => {
    router.push(`/Community/${SwitchCategory(Category)}/${PostId}`);
  };

  const chunkArray = (array, size) => {
    return array.reduce((acc, _, index) => {
      if (index % size === 0) {
        acc.push(array.slice(index, index + size));
      }
      return acc;
    }, []);
  };

  const chunkedPosts = chunkArray(userPosts, 3);

  const SwitchCategory = (Category) => {
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



  return (
    <main className="flex-col w-full h-auto justify-center items-center mt-5">
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
                    <h1 className="text-2xl font-bold">{userProfile?.userProfile?.name}</h1>
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
      {chunkedPosts.map((row, rowIndex) => (
        <div key={rowIndex} className="flex justify-start space-x-20 mb-5">
          {row.map((post) => (
            <div
              key={post.post_id}
              className="flex flex-col cursor-pointer w-[250px] h-[300px] border rounded"
            >
              <div className="flex w-full h-[250px] justify-center items-center">
                <div className="relative w-full h-full">
                  <Image
                    src={`${process.env.Localhost}${post.thumbnail.image_id}`}
                    alt="Post Image"
                    layout="fill"
                    objectFit="cover"
                    onClick={() => PostClick(post.post_id, post.category)}
                    priority
                  />
                </div>
              </div>
              <div className="flex ml-1">
                <div className="flex items-center justify-center font-bold text-sm mr-1">{`[${post.category}]`}</div>
                <div className="flex items-center justify-center font-bold text-sm">
                  {post.title}
                </div>
              </div>
              <div className="flex items-center w-full">
                <MyProfile
                  UserName={post.author.name}
                  UserProfile={post.author.image}
                />
                <div className="flex w-2/3 justify-end items-center">
                  <div className="flex items-center">
                    <div className="w-[20px]">
                      <Image
                        src="/Heart.PNG"
                        alt="Heart Image"
                        width={100}
                        height={100}
                        style={{ width: "auto", height: "auto" }}
                        priority
                      />
                    </div>
                    <div className="ml-1 text-xs">{post.like_count}</div>
                  </div>
                  <div className="flex justify-center items-center ml-1">
                    <div className="w-[17px] ">
                      <Image
                        src="/commenticon.png"
                        alt="commenticon Image"
                        width={100}
                        height={100}
                        style={{ width: "auto", height: "auto" }}
                        priority
                      />
                    </div>
                    <div className="mx-1 text-xs">{post.comment_count}</div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ))}
    </main>
  );
}

export default User;
