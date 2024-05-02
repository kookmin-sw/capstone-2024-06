import Image from "next/image";
import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from 'next/navigation';
import { useSession } from "next-auth/react";
import MyProfile from "./MyProfile";

const User = ({ }) => {
  const { data: session } = useSession();

  const [userProfile, setUserProfile] = useState(null);
  const [userPosts, setUserPosts] = useState([]);

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

  useEffect(() => {
    const fetchUserPosts = async () => {
      try {
        const response = await fetch(`${process.env.Localhost}/community/post/search?author_id=${user_id}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (response.ok) {
          const data = await response.json();
          setUserPosts(data);
        } else {
          console.error('Failed to fetch user posts');
        }
      } catch (error) {
        console.error('Error fetching user posts:', error);
      }
    };

    fetchUserPosts();
  }, []);


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

  const router = useRouter();

  const PostClick = (PostId: number, Category: string) => {
    router.push(`/Community/${SwitchCategory(Category)}/${PostId}`);
  };

  const chunkArray = (array: any[], size: number) => {
    return array.reduce((acc, _, index) => {
      if (index % size === 0) {
        acc.push(array.slice(index, index + size));
      }
      return acc;
    }, []);
  };

  const chunkedPosts = chunkArray(userPosts, 3);

  const SwitchCategory = (category: any) => {
    switch (category) {
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

  const FollowerClick = () => {
    router.push("/Users/follwer")
  }

  const FollowingClick = () => {
    router.push("/Users/following")
  }


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
                    <h1 className="text-sm hover:text-[#F4A460]">팔로워 {userProfile?.follower_count}</h1>
                  </div>
                  <div className="flex items-center mt-2">
                    <h1 className="text-sm hover:text-[#F4A460]">팔로잉 {userProfile?.followee_count}</h1>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="border-t border-gray-500 w-full mt-4"></div>
      <div className="border-t border-transparent w-full mt-4"></div>
      {chunkedPosts.map((row: any[], rowIndex: number) => (
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
