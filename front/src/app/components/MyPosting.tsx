import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import Image from 'next/image';

// 게시물 정보를 표시하는 컴포넌트
const PostItem = ({ post }) => {
  return (
    <div className="border p-4 rounded">
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
  );
};

const MyPosting = () => {
  const { data: session, status } = useSession();
  const [userPosts, setUserPosts] = useState([]);

  useEffect(() => {
    const fetchUserPosts = async () => {
      if (status === 'authenticated') {
        try {
          const response = await fetch(`/api/posts/${session?.user?.user_id}`);
          if (response.ok) {
            const data = await response.json();
            setUserPosts(data);
          } else {
            console.error('Failed to fetch user posts');
          }
        } catch (error) {
          console.error('Error fetching user posts:', error);
        }
      }
    };

    fetchUserPosts();
  }, [session, status]);

  return (
    <>
      {userPosts.map((post) => (
        <PostItem key={post.id} post={post} />
      ))}
    </>
  );
};

export default MyPosting;
