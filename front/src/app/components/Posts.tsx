  "use client";
  import MyProfile from "./MyProfile";
  import Image from "next/image";
  import React, { useState, useEffect } from "react";
  import { useRouter } from "next/navigation";

  const Posts = () => {

    const [Posts, SetPosts] = useState([]);

    useEffect(() => {
      const PostsLoad = async () => {
        try {
          const response = await fetch(`${process.env.Localhost}/post/search`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          });
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

    const PostClick = (PostId: number) => {
      router.push(`/Community/FreePost/${PostId}`);
    };

    const chunkedPosts = chunkArray(Posts, 3);

    console.log(Posts)
    return (
      <main className="flex-col w-full h-auto justify-center items-center mt-5">
        {chunkedPosts.map((row: any[], rowIndex: number) => (
          <div key={rowIndex} className="flex justify-start space-x-20 mb-5">
            {row.map((post) => (
              <div
                key={post.post_id}
                className="flex flex-col  cursor-pointer w-auto h-auto"
              >
                <MyProfile UserName="테스트 유저" UserProfile="/Profilex2.webp"/>
                <Image
                  src="/desk4.jpg"
                  alt="Post Image"
                  width={200}
                  height={10}
                  style={{ width: "100%", height: "auto" }}
                  onClick={() => PostClick(post.post_id)}
                />
                <div className="flex items-center justify-center mt-2 font-bold text-sm">
                  {post.title}
                </div>

                <div className="mr-2 flex justify-center items-center w-full">
                  {!post.Liked && (
                    <button className="w-1/3 pl-5">
                      <svg
                        className="text-red-400 w-7 h-auto fill-current"
                        viewBox="0 0 512 512"
                      >
                        <path d="M244 84L255.1 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 0 232.4 0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84C243.1 84 244 84.01 244 84L244 84zM255.1 163.9L210.1 117.1C188.4 96.28 157.6 86.4 127.3 91.44C81.55 99.07 48 138.7 48 185.1V190.9C48 219.1 59.71 246.1 80.34 265.3L256 429.3L431.7 265.3C452.3 246.1 464 219.1 464 190.9V185.1C464 138.7 430.4 99.07 384.7 91.44C354.4 86.4 323.6 96.28 301.9 117.1L255.1 163.9z" />
                      </svg>
                    </button>
                  )}
                  {post.Liked && (
                    <button className="w-1/3 pl-5">
                      <svg
                        className="text-red-400 w-7 h-auto fill-current"
                        viewBox="0 0 512 512"
                      >
                        <path d="M0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84.02L256 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 .0003 232.4 .0003 190.9L0 190.9z" />
                      </svg>
                    </button>
                  )}
                  <div className="flex justify-center items-center w-1/3">
                    <Image
                      src="/commenticon.png"
                      alt="commenticon Image"
                      width={25}
                      height={25}
                    />
                    <div className="ml-2 text-xs">{post.comment_count}</div>
                  </div>
                  <div className="flex justify-center items-center w-1/3">
                    <Image
                      src="/views.png"
                      alt="views Image"
                      width={45}
                      height={45}
                    />
                    <div className="text-xs">{post.view_count}</div>
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
