"use client";
import { useState } from 'react';
import Link from 'next/link';
import handleSubmit from '../api/user/findAccount/route';
import style from './findAccountStyle.module.css';

const FindAccountPage = () => {
  const [email, setEmail] = useState<string>('');
  const [foundUserId, setFoundUserId] = useState<string>('');
  const [error, setError] = useState<string>('');

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const data = await handleSubmit(email);
      setFoundUserId(data.user_id);
      setError('');
    } catch (error) {
      setError(error.message);
      setFoundUserId('');
    }
  };

  return (
    <form onSubmit={onSubmit} className={style.desktop}>
      <div className={style.title}>what_desk</div>
      <div className={style.subtitle}>아이디 찾기</div>
      <div className={style.rectangle62}>
        <input
          type="email"
          placeholder='이메일'
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className={style.inputField62} />
      </div>
      {error && (
        <div className={style.errormessage62}>{error}</div>
      )}
      {foundUserId && (
        <div className={style.successMessage}>찾은 계정의 아이디: {foundUserId}</div>
      )}
      <button type="submit" className={style.rectangle61}>
        <div className={style.text}>아이디 찾기</div>
      </button>
      <Link href="/api/auth/signin">
        로그인
      </Link>
    </form>
  );
};

export default FindAccountPage;
