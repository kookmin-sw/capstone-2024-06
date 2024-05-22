"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from "next/image";
import Nav from "../components/Nav";
import MyProfile from "../components/MyProfile";
import MyPosting from "../components/MyPosting";
import MyPageProfile from "../components/MyPageProfile";
import { Session } from 'next-auth';

interface ExtendedSession extends Session {
  user?: {
    name?: string | null;
    email?: string | null;
    image?: string | null;
    user_id: string;
  }
}

const MyPost = () => {
  const { data: session, status } = useSession(); // status 가져오기

  const [userPosts, setUserPosts] = useState([]);

  useEffect(() => {
    const fetchUserPosts = async () => {
      try {
        if (status === 'authenticated') { // 인증된 경우에만 게시물 가져오기
          const response = await fetch(`${process.env.Localhost}/community/post/search?author_id=${(session as ExtendedSession)?.user?.user_id}`, {
            method: 'GET',
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              'Content-Type': 'application/json',
            },
          });
          if (response.ok) {
            const data = await response.json();
            setUserPosts(data);
          } else {
            console.error('Failed to fetch user posts');
          }
        }
      } catch (error) {
        console.error('Error fetching user posts:', error);
      }
    };

    fetchUserPosts();
  }, [session, status]);

  const router = useRouter();

  const PostClick = (PostId: number, Category: string) => {
    console.log(PostId, Category)
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

  const EditpostClick = () => {
    router.push("/Mypage/EditPost");
  }

  return (
    <>
      <main className="flex-col w-full h-auto justify-center items-center mt-5">
        {chunkedPosts.map((row: any[], rowIndex: number) => (
          <div key={rowIndex} className="flex justify-start space-x-20 mb-5">
            {row.map((post) => (
              <div
                key={post.post_id}
                className="flex flex-col cursor-pointer w-[250px] h-[300px] border rounded"
              >
                <div className="flex w-full h-[250px]  justify-center items-center">
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
    </>
  );
};

export default MyPost;