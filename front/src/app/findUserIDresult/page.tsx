"use client"
import { useState } from 'react';
import Link from 'next/link';
import handleSubmit from '../api/user/findUserID/route';
import style from './findUserIDStyle.module.css';
import Nav from '../components/Nav';

const FindUserIDresult = () => {


  return (
    <>
      <Nav />
      <form onSubmit={onSubmit} className={style.desktop}>
        <div className={style.Desktop}></div>
        <div>
          <button className={style.findID} onClick={handlePasswordButtonClick}>비밀번호 찾기</button>
        </div>
        <div>
          <button className={style.findPW} onClick={handleEmailButtonClick}>이메일 찾기</button>
        </div>
        <div className={style.logo}>어떤대스크</div>
        <div className={style.line13}></div>
        <div className={style.letter}>회원님의 아이디는</div>
        <div className={style.nickname}>닉네임</div>
        <div className={style.emailAddress}>이메일주소</div>
        <div className={style.Rectangle103}></div>
        <input
          type="text"
          placeholder='nickname'
          value={nickname}
          onChange={(e) => setNickname(e.target.value)}
          className={style.inputField105} />
          </div>
        <div className={style.Rectangle106}>
        <input
          type="email"
          placeholder='email'
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className={style.inputField106} />
          </div>
        <div className={style.rectangle56}></div>
        {error && (
          <div className={style.errorMessage62}>{error}</div>
        )}
        {foundUserId && (
          <div className={style.successMessage}>등록된 아이디: {foundUserId}</div>
        )}
        <button type="submit" className={style.rectangle61}>
          <div className={style.text}>확인</div>
        </button>
        <Link href="/sign-in">
          로그인
        </Link>
      </form>
    </>
  );
};

export default FindUserIDresult;
