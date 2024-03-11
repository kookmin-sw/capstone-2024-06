"use client";
import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Image from "next/image";

const Comment = () => {

  const Postid = useParams();
  const [Comments, SetComments] = useState([{
    post_id: 0,
    parent_comment_id: 0,
    content: "string",
    comment_id: 0,
    author_id: "string",
    created_at: "2024-03-09T05:18:36.965Z",
    child_comments: [
      "string"
    ]
  }]);

  useEffect(() => {
    const CommentLoad = async () => {
      try {
        const response = await fetch(
          `${process.env.Localhost}/comment/${Postid.FreePostId}`,
          {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          }
        );
        const data = await response.json();
        SetComments(data);
        console.log(data)
      } catch (error) {
        console.error("Error", error);
      }
    };
    CommentLoad();
  }, [Postid]);

 
  const [Comment, SetComment] = useState("");

  const CommentValue = (e: {
    target: { value: React.SetStateAction<string> };
  }) => {
    SetComment(e.target.value);
  };

  const AddComment = () => {
  };


  const [Reply, SetReply] = useState("");
  const [CheckReply, SetCheckReply] = useState(true);

  const CheckReplyBt = () => {
    SetCheckReply(!CheckReply);
  };

  const ReplyValue = (e: {
    target: { value: React.SetStateAction<string> };
  }) => {
    SetReply(e.target.value);
  };

  const EnterKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      AddComment();
    }
  };

  return (
    <main className="flex-col w-full mt-4">
      <div className="flex items-center mb-3">
        <div className="text-xl font-bold mr-2">댓글</div>
        <div>{Comments.length}</div>
      </div>
      <textarea
        rows={3}
        cols={1}
        placeholder="댓글을 입력해주세요 !"
        value={Comment}
        onChange={CommentValue}
        onKeyDown={EnterKey}
        className="w-full border-2 rounded-lg p-2"
      ></textarea>
      {Comments.map((comments) => (
        <div key={comments.post_id} className="my-2 flex h-auto mb-8">
          <div className="pt-2 w-[50px]">
            <Image
              src="/Profilex2.webp"
              alt="Profile image"
              width={40}
              height={1}
              objectFit="cover"
              className="cursor-pointer mr-1  border-black rounded-full"
            />
          </div>
          <div className="flex-col  w-full">
            <div className="text-base font-bold mb-1">User name</div>
            <div className="text-sm font-light h-auto mb-2">
              {comments.content}
            </div>
            <div className="flex text-xs items-center text-[#666a73] h-4">
              <div className="mr-1">10시간 전 ·</div>
              <button className="mr-1" >
                좋아요
              </button>
              <div className="mr-1">(좋아요갯수 들어갈자리) ·</div>
              <div
                className="mr-1 cursor-pointer hover:border-b hover:border-[#666a73]"
                onClick={CheckReplyBt}
              >
                답글 달기
              </div>
              <button>· 삭제</button>
            </div>
            <div className="flex-col mt-2">
              {!CheckReply && (
                <textarea
                  rows={1}
                  cols={1}
                  placeholder="댓글을 입력해주세요 !"
                  value={Reply}
                  onChange={ReplyValue}
                  onKeyDown={EnterKey}
                  className="w-full border-2 rounded-lg p-2 text-sm h-[40px]"
                ></textarea>
              )}
              {comments.child_comments.map((Replys, index) => (
                <div key={index}  className="my-2 flex h-auto mb-1">
                  <div className="pt-2 w-[50px]">
                    <Image
                      src="/Profilex2.webp"
                      alt="Profile image"
                      width={40}
                      height={1}
                      objectFit="cover"
                      className="cursor-pointer mr-1  border-black rounded-full"
                    />
                  </div>
                  <div className="flex-col  w-full">
                    <div className="text-base font-bold mb-1">User name</div>
                    <div className="text-sm font-light h-auto mb-2">
                      {Replys}
                    </div>
                    <div className="flex text-xs items-center text-[#666a73]">
                      <div className="mr-1">10시간 전 ·</div>
                      <button
                        className="mr-1"
                      >
                        좋아요
                      </button>
                      <div className="mr-1">(좋아요갯수 들어갈 자리) ·</div>
                      <button>
                        삭제
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </main>
  );
};

export default Comment;
