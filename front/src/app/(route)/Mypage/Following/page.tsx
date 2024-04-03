import React from 'react';
import Nav from '../../../components/Nav';
import MyPageProfile from "../../../components/MyPageProfile";
import MyPagePosts from "../../../components/MyPagePosts";
import Following from '../../../components/Following';

const MyPost = () => {
  return (
    <>
      <main className="flex-col justify-center w-full h-full">
        <Nav />
        <div className="flex justify-center w-full h-auto">
          <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
            <MyPageProfile />
            <MyPagePosts />
          </div>
        </div>
      </main>
    </>
  );
};

export default MyPost;
