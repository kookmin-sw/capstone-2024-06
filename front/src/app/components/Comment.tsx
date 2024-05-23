"use client";
import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Image from "next/image";
import { useSession } from "next-auth/react";
import { useRouter, useSearchParams } from 'next/navigation';

const Comment = ({ comment_count }: { comment_count: number }) => {
  const { data: session } = useSession();
  const router = useRouter();

  const Postid = useParams();
  const [Reply, SetReply] = useState("");
  const [Comment, SetComment] = useState("");
  const [Comments, SetComments] = useState([
    {
      post_id: 0,
      parent_comment_id: 0,
      content: "",
      comment_id: 0,
      like_count: 0,
      created_at: "",
      child_comments: [
        {
          post_id: 0,
          parent_comment_id: 0,
          content: "",
          comment_id: 0,
          like_count: 0,
          created_at: "",
          author: {
            user_id: "",
            name: "",
            email: "",
            image: "",
          },
        },
      ],
      author: {
        user_id: "",
        name: "",
        email: "",
        image: "",
      },
    },
  ]);

  const CommentLoad = async () => {
    try {
      const postIdKey = Object.keys(Postid)[0];
      const response = await fetch(
        `${process.env.Localhost}/community/comment/${Postid[postIdKey]}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
      console.log(data)
      SetComments(data);
    } catch (error) {
      console.error("Error", error);
    }
  };

  useEffect(() => {
    CommentLoad();
  }, [Postid]);

  const CommentValue = (e: {
    target: { value: React.SetStateAction<string> };
  }) => {
    SetComment(e.target.value);
  };

  const ReplyValue = (e: {
    target: { value: React.SetStateAction<string> };
  }) => {
    SetReply(e.target.value);
  };

  const EnterKey = async (
    e: React.KeyboardEvent<HTMLTextAreaElement>,
    index: number | null = null
  ) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (index !== null) {
        await AddReply(index);
        SetReply("");
        toggleReplyTextarea(index);
      } else {
        await AddComment();
        SetComment("");
      }
    }
  };

  const AddComment = async () => {
    try {
      const postIdKey = Object.keys(Postid)[0];
      const PostCommentData = {
        post_id: Postid[postIdKey],
        content: Comment,
      };
      const response = await fetch(`/api/community/comment`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(PostCommentData),
      });
      const data = await response.json();
    } catch (error) {
      console.error("Error", error);
    }
    CommentLoad();
  };

  const DeleteComment = async ({ comment_id }: { comment_id: number }) => {
    try {
      const response = await fetch(
        `${process.env.Localhost}/comment/${comment_id}`,
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
      console.log("Error", error);
    }
    CommentLoad();
  };

  const AddReply = async (index: number) => {
    try {
      const postIdKey = Object.keys(Postid)[0];
      const PostReplyData = {
        post_id: Postid[postIdKey],
        parent_comment_id: Comments[index].comment_id,
        content: Reply,
      };
      const response = await fetch(`/api/community/comment`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(PostReplyData),
      });
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Error", error);
    }
    CommentLoad();
  };

  const [showReplyTextarea, setShowReplyTextarea] = useState<Array<boolean>>(
    Comments.map(() => false)
  );

  const toggleReplyTextarea = (index: number) => {
    const newShowReplyTextarea = [...showReplyTextarea];
    newShowReplyTextarea[index] = !newShowReplyTextarea[index];
    setShowReplyTextarea(newShowReplyTextarea);
  };

  const getTimeDifference = (createdAt: string | number | Date) => {
    const createdTime = new Date(createdAt);
    const currentTime = new Date();
    const difference: number = Math.floor(
      (currentTime.getTime() - createdTime.getTime()) / 1000
    );

    if (difference < 60) {
      return "방금 전";
    } else if (difference < 3600) {
      const minutes = Math.floor(difference / 60);
      return `${minutes}분 전`;
    } else if (difference < 86400) {
      const hours = Math.floor(difference / 3600);
      return `${hours}시간 전`;
    } else {
      return "오래 전";
    }
  };

  const handleAuthorImageClick = (user_id: string) => {
    router.push(`/Users?user_id=${user_id}`);
  };

  const CommentLikeBtClick = async (comment_id : number) => {
    try {
      const response = await fetch(
        `${process.env.Localhost}/community/comment/like/${comment_id}`,
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
    <main className="flex-col w-full mt-4">
      <div className="flex items-center mb-3">
        <div className="text-xl font-bold mr-2">댓글</div>
        <div>{comment_count}</div>
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
      {Comments.map((comment, index) => (
        <div key={comment.comment_id} className="my-2 flex h-auto mb-8">
          <div className="pt-2 w-[50px]">
            <Image
              src={comment.author.image}
              alt="Profile image"
              width={40}
              height={1}
              objectFit="cover"
              className="cursor-pointer mr-1  border-black rounded-full"
              onClick={() => handleAuthorImageClick(comment.author.user_id)}
            />
          </div>
          <div className="flex-col  w-full">
            <div className="text-base font-bold mb-1">{comment.author.name}</div>
            <div className="text-sm font-light h-auto mb-2">
              {comment.content}
            </div>
            <div className="flex text-xs items-center text-[#666a73] h-4">
              <div className="mr-1">
                {getTimeDifference(comment.created_at)}
              </div>
              <button onClick={()=>CommentLikeBtClick(comment.comment_id)} className="mr-1 cursor-pointer hover:border-b hover:border-[#666a73]">
                좋아요
              </button>
              <div className="mr-1">{comment.like_count}</div>
              <div
                className="mr-1 cursor-pointer hover:border-b hover:border-[#666a73]"
                onClick={() => toggleReplyTextarea(index)}
              >
                답글 달기
              </div>
              <div className="mr-1">{comment.child_comments.length}</div>
              <button
                className="cursor-pointer hover:border-b hover:border-[#666a73]"
                onClick={() =>
                  DeleteComment({ comment_id: comment.comment_id })
                }
              >
                삭제
              </button>
            </div>
            <div className="flex-col mt-2">
              {showReplyTextarea[index] && (
                <textarea
                  rows={1}
                  cols={1}
                  placeholder="댓글을 입력해주세요 !"
                  value={Reply}
                  onChange={ReplyValue}
                  onKeyDown={(e) => EnterKey(e, index)}
                  className="w-full border-2 rounded-lg p-2 text-sm h-[40px]"
                ></textarea>
              )}
              {comment.child_comments.map((replys) => (
                <div key={replys.comment_id} className="my-2 flex h-auto mb-1">
                  <div className="pt-2 w-[50px]">
                    <Image
                      src={replys.author.image}
                      alt="Profile image"
                      width={40}
                      height={1}
                      objectFit="cover"
                      className="cursor-pointer mr-1  border-black rounded-full"
                      onClick={() => handleAuthorImageClick(comment.author.user_id)}
                    />
                  </div>
                  <div className="flex-col  w-full">
                    <div className="text-base font-bold mb-1">{replys.author.name.substring(0, 4)}</div>
                    <div className="text-sm font-light h-auto mb-2">
                      {replys.content}
                    </div>
                    <div className="flex text-xs items-center text-[#666a73] h-4">
                      <div className="mr-1">
                        {getTimeDifference(replys.created_at)}
                      </div>
                      <button onClick={()=>CommentLikeBtClick(replys.comment_id)} className="mr-1 cursor-pointer hover:border-b hover:border-[#666a73]">
                        좋아요
                      </button>
                      <div className="mr-1">{replys.like_count}</div>
                      <button
                        className="cursor-pointer hover:border-b hover:border-[#666a73]"
                        onClick={() =>
                          DeleteComment({ comment_id: replys.comment_id })
                        }
                      >
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
