"use client";
import { useEffect, useState } from 'react';
import { useRouter } from "next/navigation";
import { useSession } from 'next-auth/react';
import Image from 'next/image';

const MyPosting = () => {
  const { data: session, status } = useSession();
  const [userPosts, setUserPosts] = useState([]);

  useEffect(() => {
    const fetchUserPosts = async () => {
      try {
        if (status === 'authenticated') {
          const response = await fetch(`/api/posts/${session?.user?.user_id}`, {
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

  const PostClick = (PostId: number) => {
    router.push(`/Community/Post/${PostId}`);
  };

  return (
    <>
      {userPosts.map((post) => (
        <div key={post.id} className="border p-4 rounded" onClick={() => PostClick(post.id)}>
          <div className="absolute w-[1032px] h-[182px] left-[189px] top-[224px] border border-gray-300" />
          <div className="absolute left-[206px] top-[235px] font-semibold text-base leading-9 text-black">
            {post.title}
          </div>
          <div className="absolute w-[1205px] h-[173px] left-[209px] top-[285px] font-thin text-lg leading-[39px] text-black">
            {post.content}
          </div>
          <div className="absolute left-[206px] top-[355px] ">
            <Image
              src="/Heart.png"
              alt="Heart image"
              width={20}
              height={20}
              className="cursor-pointer mr-1 rounded-full"
            />
          </div>
          <div className="absolute left-[236px] top-[355px] ">
            <Image
              src="/Comments.png"
              alt="Comments image"
              width={22}
              height={22}
              className="cursor-pointer mr-1 rounded-full"
            />
          </div>
        </div>
      ))}
    </>
  );
};

export default MyPosting;
