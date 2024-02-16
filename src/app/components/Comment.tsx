"use client";

import React, { useState } from "react";

const Comment = () => {
  const [comments, setComments] = useState([
    { id: 1, text: "테스트 1", likes: 0 },
    { id: 2, text: "테스트 2", likes: 0 },
    // 필요한 만큼 댓글을 추가할 수 있습니다.
  ]);

  const handleLike = (id:number) => {
    setComments((prevComments) =>
      prevComments.map((comment) =>
        comment.id === id ? { ...comment, likes: comment.likes + 1 } : comment
      )
    );
  };

  return (
    <main className="flex-col w-full">
      <div className="text-xl font-bold mt-4">댓글</div>
      {comments.map((comment) => (
        <div key={comment.id} className="mb-2 border">
          <div>{comment.text}</div>
          <button onClick={() => handleLike(comment.id)}>
            좋아요 {comment.likes}
          </button>
        </div>
      ))}
    </main>
  );
};

export default Comment;

