"use client";
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from 'next/image';
import MyProfile from './MyProfile';

const MyScrappedPosts = () => {
  const { data: session, status } = useSession();
  const [scrappedPosts, setScrappedPosts] = useState([]);

  useEffect(() => {
    const fetchScrappedPosts = async () => {
      try {
        if (status === 'authenticated') {
          const response = await fetch(`/what-desk-api/user/scrapped_post`, {
            method: 'GET',
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              'Content-Type': 'application/json',
            },
          });
          if (response.ok) {
            const data = await response.json();
            setScrappedPosts(data);
          } else {
            console.error('Failed to fetch scrapped posts');
          }
        }
      } catch (error) {
        console.error('Error fetching scrapped posts:', error);
      }
    };

    fetchScrappedPosts();
  }, [session, status]);

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

  const chunkedPosts = chunkArray(scrappedPosts, 3);

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

  return (
    <>
      <div className="relative w-screen h-screen bg-[background-color]">
        <div className="absolute w-[251px] h-[91px] left-[200px] top-[66px] font-inter font-semibold text-4xl leading-14 text-yellow-600">
          어떤데스크
        </div>
        <div className="absolute w-[173px] h-[83px] left-[200px] top-[144px] font-semibold text-base leading-9 text-black">
          스크랩
        </div>
        <div className="absolute w-[1049px] h-0 left-[179px] top-[184px] border border-gray-300 transform rotate-0.05" />
        <div className="absolute left-[200px] top-[210px]">
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
        </div>
      </div>
    </>
  );

};

export default MyScrappedPosts;
