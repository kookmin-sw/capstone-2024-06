import React from 'react';
import Nav from '../../../components/Nav';
import UserFollowerlist from '../../../components/UserFollower';

const UserFollower = () => {
  return (
    <>
      <main className="flex-col justify-center w-full h-full">
        <Nav />
        <div className="flex justify-center w-full h-auto">
          <div className="flex items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">

            <UserFollowerlist />
          </div>
        </div>
      </main>
    </>
  );
};

export default UserFollower;