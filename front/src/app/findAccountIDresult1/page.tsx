"use client"
import { useState } from 'react';
import { useRouter} from "next/navigation";

import Link from 'next/link';
import handleSubmit from '../api/user/findAccountIDresult1/route';
import style from './findAccountIDresult1Style.module.css';
import Nav from '../components/Nav';



const findUserIDresult = () => {

  
const router = useRouter();
const handlePasswordButtonClick = () => {
  router.push('/findAccountID');
  // 비밀번호 찾기 버튼을 클릭했을 때의 동작을 여기에 작성합니다.
  // 예: 비밀번호 찾기 모달을 열거나 다른 페이지로 이동합니다.
};

const handleEmailButtonClick = () => {
  console.log('이메일 찾기 버튼이 클릭되었습니다.');
  router.push('/findUserID');
  // 여기에 이메일 찾기와 관련된 추가적인 로직을 작성하세요.
};
// DESKTOP33
  return (
    <>
    <Nav/>
    <div>
    <div className={style.Desktop}></div>
    <div>
          <button className={style.findID} onClick={handlePasswordButtonClick}>비밀번호 찾기</button>
        </div>
        <div>
          <button className={style.findPW} onClick={handleEmailButtonClick}>아이디 찾기</button>
        </div>
    <div className={style.topbar}></div>
    <div className={style.Rectangle103}></div>
    <div className={style.logo}>어떤대스크</div>
    <div className={style.notice}>회원님의 비밀번호는 ****입니다.</div>
    <div className={style.line13}></div>


  </div>
  </>
  )
};

export default findUserIDresult;
