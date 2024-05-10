"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

import Nav from "../components/Nav";
import style from "./findAccountIDStyle.module.css";

const FindUserIDResult = () => {
  const [userID, setUserID] = useState("");
  const router = useRouter();
  
  const handleSubmit = async (
    user_id: string,
    setError: (error: string) => void
  ) => {
    
    try {
      const response = await fetch(`${process.env.Localhost}/user/password}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id}),
      });
      const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        await handleSubmit(user_id, setError);
      };
  
      if (response.ok) {
        // 회원가입 성공
        console.log('비번찾기성공.');
      } else {
        // 회원가입 실패
        setError('실패했습니다.');
      }
    } catch (error) {
      console.error('오류 발생:', error);
      setError('오류가 발생했습니다.');
    }
  };




  const handlePasswordButtonClick = () => {
    router.push("/findAccountID");
  };

  const handleEmailButtonClick = () => {
    router.push("/findUserID");
  };

  const handleNavigate = () => {
    router.push("/nextPage"); // 다른 페이지로 이동
  };

  const nextClick = () => {
    router.push("/findAccountIDresult");
  };

  return (
    <>
      <Nav />
        <form onSubmit={handleSubmit} className={style.desktop}> {/* 클릭 이벤트로 페이지 전환 */}
        <div className={style.logo}>어떤데스크</div>
        <div className={style.box}></div>
        <div className={style.text}>
          비밀번호를 찾고자 하는 아이디를 입력해주세요
        </div>
        <div className={style.smallbox}>
          <input
            type="text"
            value={userID} // 입력된 사용자 ID
            onChange={(e) => setUserID(e.target.value)} // 사용자 ID 값 설정
            placeholder="ID를 입력하세요"
            className={style.inputField} // 스타일 클래스
          />
        </div>
        <div className={style.colorbox}></div>
        <div className={style.smallbox1}></div>
        <div className={style.inputbox}></div>

        <div>
        <button type="submit" className={style.colorboxtext}>
            다음
          </button>
        </div>
        <div className={style.line}></div>
        <div>
          <button className={style.findID} onClick={handlePasswordButtonClick}>
            비밀번호 찾기
          </button>
        </div>
        <div>
          <button
            className={style.findPW}
            onClick={handleEmailButtonClick}
          >
            아이디 찾기
          </button>
        </div>
        </form>
    </>
  );
};

export default FindUserIDResult;