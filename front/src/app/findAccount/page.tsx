// desktop21
"use client";
import { useState } from 'react';
import { useRouter} from "next/navigation";
import style from './findAccountStyle.module.css';
import Image from 'next/image';
import Link from 'next/link';
import Nav from '../components/Nav';


const FindAccountPage = () => {
   
  const router = useRouter();

  const handlePasswordButtonClick = () => {
    console.log('비밀번호 찾기 버튼이 클릭되었습니다.');
    router.push("/findAccountID");
    // 여기에 비밀번호 찾기와 관련된 추가적인 로직을 작성하세요.
  };

  const handleEmailButtonClick = () => {
    console.log('이메일 찾기 버튼이 클릭되었습니다.');
    router.push("/findUserID");
    // 여기에 이메일 찾기와 관련된 추가적인 로직을 작성하세요.
  };


  return (
    <>
      <Nav />
     
    <div className={style.desktop}>
      <div className={style.top}>어떤데스크</div>
      <div>
          <button className={style.findID} onClick={handlePasswordButtonClick}>비밀번호 찾기</button>
        </div>
        <div>
          <button className={style.findPW} onClick={handleEmailButtonClick}>아이디 찾기</button>
        </div>
      
      <div>
        <button className={style.Rectangle104} onClick={handlePasswordButtonClick}>
        <Image 
                src="/arcticons_password.png"
                alt="arcticons_password.png"
                width={100}
                height={100}
              /><div className="absolute left-[180px] top-[30px]">비밀번호 찾기</div></button>
      </div>
      
      <div>
        <button className={style.Rectangle103} onClick={handleEmailButtonClick}>
        <Image 
                src="/et_profile-male.png"
                alt="et_profile-male.png"
                width={90}
                height={80}
              />
              <div className="absolute left-[180px] top-[30px]">아이디 찾기</div></button>
      </div>
      <div className={style.line}></div> {/* 구분선 */}

    </div>
    </>
  );
};

export default FindAccountPage;
