"use client";
import { useState } from 'react';
import Link from 'next/link';
import handleSubmit from '../api/user/sing-up/route';
import style from './signupStyle.module.css';
import Nav from '../components/Nav';

const RegisterForm = () => {


  const [user_id, setUserid] = useState<string>('');
  const [name, setUsername] = useState<string>('');
  const [email, setEmail] = useState<string>('');
  const image = '/Profilex2.webp';
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [error, setError] = useState<string>('');

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    await handleSubmit(user_id, name, email, image, password, confirmPassword, setError);
  };

  return (
    <>
      <Nav />
      <form onSubmit={onSubmit} className={style.desktop}>
        <div className={style.title}>what_desk</div>
        <div className={style.subtitle}>회원가입</div>
        <div className={style.rectangle63}>
          <input
            type="text"
            placeholder='닉네임'
            value={name}
            onChange={(e) => setUsername(e.target.value)}
            className={style.inputField63} />
        </div>
        <div className={style.rectangle64}>
          <input
            type="text"
            placeholder='아이디'
            value={user_id}
            onChange={(e) => setUserid(e.target.value)}
            className={style.inputField64} />
        </div>
        <div className={style.rectangle58}>
          <input
            type="password"
            placeholder="비밀번호"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={style.inputField58} />
        </div>
        <div className={style.rectangle59}>
          <input
            type="password"
            placeholder="비밀번호 확인"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className={style.inputField59} />
        </div>
        {error && (
          <div className={style.errormessage59}>
            {error}
          </div>
        )}
        <div className={style.rectangle62}>
          <input
            type="email"
            placeholder='이메일'
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={style.inputField62} />
        </div>
        <div className={style.line4}></div>
        <button type="submit" className={style.rectangle55}>
          회원가입하기</button>
        <div className={style.loginbutton}>로그인</div>
        <div>
          <Link href="/sign-in">
            <span className={style.loginbutton}>로그인</span>
          </Link>
        </div>
        <div>
          <Link href="/findAccount">
            <span className={style.FindAccount}>계정찾기</span>
          </Link>
        </div>
      </form>
    </>
  );
};

export default RegisterForm;


