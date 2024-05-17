"use client";
import Link from 'next/link';
import Nav from '../../components/Nav';

const FindAccountPage = () => {
  return (
    <>
      <Nav />
      <div className="flex flex-col items-center justify-center w-full h-full bg-[background-color] pt-20">
        <div className="font-inter font-semibold text-4xl leading-14 text-yellow-700 text-center">
          what_desk
        </div>
        <div className="w-[173px] h-[83px] font-semibold text-base leading-9 text-black text-center mt-4">
          아이디/비밀번호 찾기
        </div>
        <div className="flex space-x-4 mt-4">
          <Link href="/findAccount/findUserID">
            <button className="px-4 py-2 bg-yellow-500 text-white font-semibold rounded hover:bg-yellow-600 focus:outline-none">
              아이디 찾기
            </button>
          </Link>
          <Link href="/findAccount/findPassword">
            <button className="px-4 py-2 bg-yellow-500 text-white font-semibold rounded hover:bg-yellow-600 focus:outline-none">
              비밀번호 찾기
            </button>
          </Link>
        </div>
      </div>
    </>
  );
}

export default FindAccountPage;
