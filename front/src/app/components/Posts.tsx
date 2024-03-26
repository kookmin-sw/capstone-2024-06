"use client";
import MyProfile from "./MyProfile";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

const Posts = ({ PostCateGory }: { PostCateGory: string }) => {
  const [Posts, SetPosts] = useState([]);

  useEffect(() => {
    const PostsLoad = async () => {
      try {
        const response = await fetch(
          `${process.env.Localhost}/post/search?category=${PostCateGory}`,
          {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          }
        );
        const data = await response.json();
        SetPosts(data);
      } catch (error) {
        console.error("Error", error);
      }
    };
    PostsLoad();
  }, []);

  const router = useRouter();

  const chunkArray = (array: any[], size: number) => {
    return array.reduce((acc, _, index) => {
      if (index % size === 0) {
        acc.push(array.slice(index, index + size));
      }
      return acc;
    }, []);
  };

  const PostClick = (PostId: number, Category: string) => {
    router.push(`/Community/${SwitchCategory(Category)}/${PostId}`);
  };

  const chunkedPosts = chunkArray(Posts, 3);

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
    <main className="flex-col w-full h-auto justify-center items-center mt-5">
      {chunkedPosts.map((row: any[], rowIndex: number) => (
        <div key={rowIndex} className="flex justify-start space-x-20 mb-5">
          {row.map((post) => (
            <div
              key={post.post_id}
              className="flex flex-col  cursor-pointer w-[400px] h-[300px] border rounded"
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
  );
};

export default Posts;
