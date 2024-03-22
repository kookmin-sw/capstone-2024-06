
"use client";
import { useState } from 'react';
import style from './findAccountStyle.module.css';
import Link from 'next/link';
import Nav from '../components/Nav';


const FindAccountPage = () => {
  const handlePasswordButtonClick = () => {
    console.log('비밀번호 찾기 버튼이 클릭되었습니다.');
    // 여기에 비밀번호 찾기와 관련된 추가적인 로직을 작성하세요.
  };

  const handleEmailButtonClick = () => {
    console.log('이메일 찾기 버튼이 클릭되었습니다.');
    // 여기에 이메일 찾기와 관련된 추가적인 로직을 작성하세요.
  };

  return (
    <>
      <Nav />
    <div className={style.desktop}>
      <div className={style.top}>어떤데스크</div>
      
      <div>
        <button className={style.Rectangle104} onClick={handlePasswordButtonClick}>비밀번호 찾기</button>
      </div>
      
      <div>
        <button className={style.Rectangle103} onClick={handleEmailButtonClick}>이메일 찾기</button>
      </div>
    </div>
    </>
  );
};

export default FindAccountPage;
