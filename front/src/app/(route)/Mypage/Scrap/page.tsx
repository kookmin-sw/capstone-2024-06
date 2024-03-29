import React from 'react';
import Image from 'next/image';
import Nav from '../../../components/Nav';
import MyScrappedPosts from '../../../components/Scrap';

const MyPost = () => {
  return (
    <>
      <Nav />
      <MyScrappedPosts />
    </>
  );
};

export default MyPost;
