"use client";
import { useState } from 'react';
import Link from 'next/link';
import handleSubmit from '../api/user/findUserID/route';
import style from './findUserIDStyle.module.css';
import Nav from '../components/Nav';

const FindUserIDpage = () => {
  const [email, setEmail] = useState<string>('');
  // 찾은 사용자의 아이디를 저장하는 상태 변수
  const [foundUserId, setFoundUserId] = useState<string>('');
  // 에러 메시지를 관리하는 상태 변수
  const [error, setError] = useState<string>('');

  // 폼 제출 핸들러
  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      // 이메일을 사용하여 사용자 정보를 찾는 함수 호출
      const data = await handleSubmit(email, setFoundUserId, setError);
      // 사용자 정보를 찾으면 찾은 사용자의 아이디를 설정하고 에러 메시지 초기화
      setFoundUserId(data.user_id);
      setError('');
    } catch (error) {
      // 에러가 발생하면 에러 메시지를 설정하고 찾은 사용자의 아이디 초기화
      setError(error.message);
      setFoundUserId('');
    }
  };

  return (
    <>
      <Nav />
      <form onSubmit={onSubmit} className={style.desktop}>
        <div className={style.title}>what_desk</div>
        <div className={style.subtitle}>아이디 찾기</div>
        <div className={style.Rectangle103}></div>
        <div className={style.Rectangle104}></div>
        <div className={style.Desktop}></div>
        <div className={style.topbar}></div>
        <div className={style.name}></div>
        <div className={style.idFinder}></div>
        <div className={style.passwordFinder}></div>
        <div className={style.line13}></div>
        <div className={style.registeredEmail}>회원정보에 등록된 이메일</div>
        <div className={style.nickname}>닉네임</div>
        <div className={style.emailAddress}>이메일주소</div>
        <div className={style.Rectangle105}></div>
        <div className={style.Rectangle106}></div>
        


        
        <div className={style.rectangle56}>
          {/* 이메일 입력 필드 */}
          <input
            type="email"
            placeholder='이메일'
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={style.inputField56} />
        </div>
        {/* 에러 메시지 출력 */}
        {error && (
          <div className={style.errormessage62}>{error}</div>
        )}
        {/* 찾은 사용자의 아이디 출력 */}
        {foundUserId && (
          <div className={style.successMessage}>등록된 아이디: {foundUserId}</div>
        )}
        {/* 아이디 찾기 버튼 */}
        <button type="submit" className={style.rectangle61}>
          <div className={style.text}>확인</div>
        </button>
        {/* 로그인 링크 */}
        <Link href="/sign-in">
          로그인
        </Link>
      </form>
    </>
  );
};

export default FindUserIDpage;
