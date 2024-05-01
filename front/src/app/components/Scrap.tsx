"use client";
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from 'next/image';

const MyScrappedPosts = () => {
  const { data: session, status } = useSession();
  const [scrappedPosts, setScrappedPosts] = useState([]);

  useEffect(() => {
    const fetchScrappedPosts = async () => {
      try {
        if (status === 'authenticated') {
          const response = await fetch(`${process.env.Localhost}/community/post/scrap/{post_id} `, {
            method: 'POST',
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

  const PostClick = (PostId: number) => {
    router.push(`/Community/${PostId}`);
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
        {scrappedPosts.map((post) => (
          <div key={post.id} className="absolute w-[1032px] left-[189px] top-[224px] border border-gray-300" onClick={() => PostClick(post.id)}>
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
      </div>
    </>
  );

};

export default MyScrappedPosts;
