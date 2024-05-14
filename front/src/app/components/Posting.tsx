"use client";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Comment from "./Comment";
import { useSession } from "next-auth/react";

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
      image: "",
    },
    scrapped: true,
    liked: true,
    images: [
      {
        image_id: "",
        filename: "",
      },
    ],
    content: "string",
  });

  useEffect(() => {
    if (!session) return;
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
        console.log(data);
        SetPosting(data);
      } catch (error) {
        console.error("Error", error);
      }
    };
    PostLoadBt();
  }, [Postid, session]);

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
    if (user_id === session?.user?.user_id) {
      router.push("/Mypage");
    } else router.push(`/Users?user_id=${user_id}`);
    // Users 페이지로 이동
  };

  const HeartBtClick = async () => {
    try {
      const postIdKey = Object.keys(Postid)[0];

      const response = await fetch(
        `${process.env.Localhost}/community/post/like/${Postid[postIdKey]}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Error", error);
    }
  };

  const ScrapBtClick = async () => {
    try {
      const postIdKey = Object.keys(Postid)[0];

      const response = await fetch(
        `${process.env.Localhost}/community/post/scrap/${Postid[postIdKey]}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Error", error);
    }
  };

  return (
    <main className="flex">
      <div className="flex-col w-[900px] h-auto mr-2">
        <div className="w-full flex mb-1">
          <div className="text-xl font-bold pl-3 w-full ">{Posting.title}</div>
          <div className="mr-2 w-[120px]">
            {Posting.created_at.slice(0, 10)}
          </div>
          <div className="">{Posting.created_at.slice(11, 16)}</div>
        </div>
        <div className="w-full border-b flex items-center pb-1">
          <Image
            src={Posting.author.image}
            width={40}
            height={30}
            alt={""}
            className="rounded-full"
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
        <div className="flex justify-center items-center w-full m-6">
        <button
          onClick={PostingDeleteBt}
          className="bg-transparent w-[200px] h-[50px] hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
        >
          글 삭제하기
        </button></div>
      </div>
      <div className="flex-col w-[100px] border-2 rounded-full h-fit sticky top-5 pt-5 ml-2">
        <div className="flex justify-center items-center">
          {Posting.liked ? (
            <div
              onClick={HeartBtClick}
              className="flex cursor-pointer justify-center items-center border-2 rounded-full h-[50px] w-[50px]"
            >
              <div className="w-[25px]">
                <Image
                  src="/Heart2.png"
                  alt="commenticon Image"
                  width={1000}
                  height={1000}
                />
              </div>
            </div>
          ) : (
            <div
              onClick={HeartBtClick}
              className="flex cursor-pointer justify-center items-center border-2 rounded-full h-[50px] w-[50px]"
            >
              <div className="w-[25px]">
                <Image
                  src="/Heart1.png"
                  alt="commenticon Image"
                  width={1000}
                  height={1000}
                />
              </div>
            </div>
          )}
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
        <div className="flex justify-center items-center">
          {Posting.scrapped ? (
            <div
              onClick={ScrapBtClick}
              className="flex cursor-pointer justify-center items-center border-2 rounded-full h-[50px] w-[50px]"
            >
              <div className="w-[17px]">
                <Image
                  src="/scrap2.png"
                  alt="commenticon Image"
                  width={1000}
                  height={1000}
                />
              </div>
            </div>
          ) : (
            <div
              onClick={ScrapBtClick}
              className="flex cursor-pointer justify-center items-center border-2 rounded-full h-[50px] w-[50px]"
            >
              <div className="w-[17px]">
                <Image
                  src="/scrap1.png"
                  alt="commenticon Image"
                  width={1000}
                  height={1000}
                />
              </div>
            </div>
          )}
        </div>
        <div className="flex justify-center items-center">
          {Posting.scrap_count}
        </div>
      </div>
    </main>
  );
};

export default Posting;
