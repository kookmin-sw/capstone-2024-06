"use client";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Comment from "./Comment";
import { useSession } from "next-auth/react";
import User from "./UserPage"

const Posting = () => {
  const { data: session } = useSession();
  const router = useRouter();

  const Postid = useParams();

  const [Posting, SetPosting] = useState({
    post_id: 0,
    title: "",
    category: "",
    scrap_count: 0,
    like_count: 0,
    view_count: 0,
    comment_count: 0,
    created_at: "",
    author: {
      user_id: "",
      name: "",
      email: "",
      image: ""
    },
    scrapped: true,
    liked: true,
    images: [
      {
        image_id: "",
        filename: ""
      }
    ],
    content: "string"
  });

  useEffect(() => {
    const PostLoadBt = async () => {
      try {
        const postIdKey = Object.keys(Postid)[0];
        const response = await fetch(
          `${process.env.Localhost}/community/post/${Postid[postIdKey]}`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
              "Content-Type": "application/json",
            },
          }
        );
        const data = await response.json();
        SetPosting(data);
      } catch (error) {
        console.error("Error", error);
      }
    };
    PostLoadBt();
  }, [Postid]);

  const PostingDeleteBt = async () => {
    try {
      const postIdKey = Object.keys(Postid)[0];
      const response = await fetch(
        `${process.env.Localhost}/post/${Postid[postIdKey]}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
    } catch (error) {
      console.error("Error", error);
    }
    router.push("/Community");
  };

  const handleAuthorImageClick = (user_id: string) => {
    router.push(`/Users?user_id=${user_id}`);
    // Users 페이지로 이동
  };

  return (
    <main className="flex">
      <div className="flex-col w-[900px] h-auto mr-2">
        <div className="w-full flex mb-1">
          <div className="text-xl font-bold pl-3 w-full ">{Posting.title}</div>
          <div className="mr-2 w-[120px]">{Posting.created_at.slice(0, 10)}</div>
          <div className="">{Posting.created_at.slice(11, 16)}</div>
        </div>
        <div className="w-full border-b flex items-center pb-1">
          <Image src={Posting.author.image} width={40} height={30} alt={""} className="rounded-full"
            onClick={() => handleAuthorImageClick(Posting.author.user_id)}
          />
          <div className="ml-2 w-full">{Posting.author.name}</div>
          <div className="flex w-[78px] h-full mr-1 text-xs">
            <div className="mr-1">조회수</div>
            <div>{Posting.view_count}</div>
          </div>
          <div className="flex w-[78px] h-full mr-1 text-xs">
            <div className="mr-1">좋아요</div>
            <div>{Posting.like_count}</div>
          </div>
          <div className="flex w-[60px] h-full text-xs">
            <div className="mr-1">댓글</div>
            <div>{Posting.comment_count}</div>
          </div>
        </div>
        <div className="flex w-full justify-center items-center my-10">
          {Posting.images && Posting.images.length > 0 && (
            <Image
              src={`${process.env.Localhost}${Posting.images[0].image_id}`}
              width={600}
              height={300}
              alt=""
            />
          )}
        </div>
        <div className="flex justify-start items-center text-lg">
          {Posting.content}
        </div>
        <Comment comment_count={Posting.comment_count} />
        <button
          onClick={PostingDeleteBt}
          className="animate-bounce bg-transparent w-[200px] h-[50px] hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
        >
          글 삭제하기
        </button>
      </div>
      <div className="flex-col w-[100px] border h-fit sticky top-5">
        <div className="flex justify-center items-center">
          <div className="flex justify-center items-center border-2 rounded-full h-[50px] w-[50px]">
            <svg
              className="text-red-400 w-7 h-auto fill-current"
              viewBox="0 0 512 512"
            >
              <path d="M244 84L255.1 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 0 232.4 0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84C243.1 84 244 84.01 244 84L244 84zM255.1 163.9L210.1 117.1C188.4 96.28 157.6 86.4 127.3 91.44C81.55 99.07 48 138.7 48 185.1V190.9C48 219.1 59.71 246.1 80.34 265.3L256 429.3L431.7 265.3C452.3 246.1 464 219.1 464 190.9V185.1C464 138.7 430.4 99.07 384.7 91.44C354.4 86.4 323.6 96.28 301.9 117.1L255.1 163.9z" />
            </svg>
          </div>
        </div>
        <div className="flex justify-center items-center">
          {Posting.like_count}
        </div>
        <div className="flex justify-center items-center">
          <div className="flex justify-center items-center border-2 rounded-full h-[50px] w-[50px]">
            <div className="w-[20px]">
              <Image
                src="/commenticon.png"
                alt="commenticon Image"
                width={1000}
                height={1000}
              />
            </div>
          </div>
        </div>
        <div className="flex justify-center items-center">
          {Posting.comment_count}
        </div>
      </div>
    </main>
  );
};

export default Posting;
