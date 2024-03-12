"use client";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Comment from "./Comment";

const Posting = () => {
  const router = useRouter();

  const Postid = useParams();

  const [Posting, SetPosting] = useState({
    author_id: 1,
    title: "",
    view_count: 0,
    like_count: 0,
    comment_count: 0,
    content: "",
    created_at: "",
    comment: "",
    author_image: "",
  });

  useEffect(() => {
    const PostLoadBt = async () => {
      try {
        const response = await fetch(
          `${process.env.Localhost}/post/${Postid.FreePostId}`,
          {
            method: "GET",
            headers: {
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
      const response = await fetch(
        `${process.env.Localhost}/post/${Postid.FreePostId}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Error", error);
    }
    router.push("/Community");
  };


  const HeartBtClick = async () => {
    try {
      const response = await fetch(
        `${process.env.Localhost}/like/post/${Postid.FreePostId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Error", error);
    }
    router.push(`http://localhost:3000/Community/FreePost/${Postid.FreePostId}`);
  };
  console.log(Posting)

  return (
    <main className="flex">
      <div className="flex-col w-[85%] h-auto">
        <div className="w-full flex mb-1">
          <div className="text-xl font-bold pl-3 w-[85%] ">{Posting.title}</div>
          <div className="mr-2">{Posting.created_at.slice(0, 10)}</div>
          <div className="">{Posting.created_at.slice(11, 16)}</div>
        </div>
        <div className="w-full border-b flex items-center pb-1">
          <Image src={Posting.author_image} width={40} height={30} alt={""} />
          <div className="ml-2 w-full">{Posting.author_id}</div>
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
          <Image src="/desk5.png" width={600} height={300} alt={""} />
        </div>
        <div className="flex justify-start items-center text-lg">
          {Posting.content}
        </div>
        <Comment comment_count={Posting.comment_count}/>
        <button
          onClick={PostingDeleteBt}
          className="animate-bounce bg-transparent w-[200px] h-[50px] hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
        >
          글 삭제하기
        </button>
      </div>
      <div className="w-[100px] border h-[300px]">
        <svg
          className="text-red-400 w-7 h-auto fill-current"
          viewBox="0 0 512 512"
          onClick={HeartBtClick}
        >
          <path d="M0 190.9V185.1C0 115.2 50.52 55.58 119.4 44.1C164.1 36.51 211.4 51.37 244 84.02L256 96L267.1 84.02C300.6 51.37 347 36.51 392.6 44.1C461.5 55.58 512 115.2 512 185.1V190.9C512 232.4 494.8 272.1 464.4 300.4L283.7 469.1C276.2 476.1 266.3 480 256 480C245.7 480 235.8 476.1 228.3 469.1L47.59 300.4C17.23 272.1 .0003 232.4 .0003 190.9L0 190.9z" />
        </svg>
      </div>
    </main>
  );
};

export default Posting;
