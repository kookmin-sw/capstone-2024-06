"use client";
import { useState } from 'react';
import { useRouter } from "next/navigation";

import Nav from '../components/Nav';
import style from './findAccountIDStyle.module.css';

const FindUserIDResult = () => {
  const [email, setEmail] = useState(''); // 사용자 이메일을 위한 상태 변수
  const router = useRouter();

  const handlePasswordButtonClick = () => {
    router.push("/findAccountID");
  };

  const handleEmailButtonClick = () => {
    router.push("/findUserID");
  };

  const nextClick = () => {
    console.log(`입력된 이메일: ${email}`); // 입력된 이메일 로그
    router.push("findAccountIDresult1");
  };

  return (
    <>
      <Nav /> {/* 네비게이션 컴포넌트 */}
      <div>
        <div className={style.logo}>어떤데스크</div> {/* 로고 */}
        <div className={style.box}></div> {/* 박스 디자인 */}
        <div className={style.text}>회원정보에 등록된 이메일로 인증</div> {/* 안내 문구 */}
        
        <div className={style.smallbox}>
          <input
            type="email" // 이메일 입력 필드
            value={email} // 입력된 이메일
            onChange={(e) => setEmail(e.target.value)} // 상태 업데이트
            placeholder="이메일을 입력하세요" // 플레이스홀더 텍스트
            className={style.inputField} // 스타일 클래스
          />
        </div>
        
        <div className={style.colorbox}></div>
        <div>
          <button className={style.colorboxtext} onClick={nextClick}>다음</button> {/* 다음 버튼 */}
        </div>
        
        <div className={style.line}></div> {/* 구분선 */}
        
        <div>
          <button className={style.findID} onClick={handlePasswordButtonClick}>비밀번호 찾기</button>
        </div>
        <div>
          <button className={style.findPW} onClick={handleEmailButtonClick}>아이디 찾기</button>
        </div>
      </div>
    </>
  );
};

export default FindUserIDResult;
