"use client";
import { useState } from 'react';
import Link from 'next/link';
import Nav from '../components/Nav';
import style from './findAccountStyle.module.css';

const FindAccountPage = () => {
  const [password, setPassword] = useState<string>('');
  const [email, setEmail] = useState<string>('');
  const [error, setError] = useState<string>('');

  return (
    <>
      <Nav />
      <form className={style.desktop}>
        <div className={style.top}>whatdesk</div>
        <div className={style.Rectangle104}>
          <input
            type="text"
            placeholder='비밀번호 찾기'
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={style.inputField104} />
        </div>
        <div className={style.top}></div>
        <div className={style.Rectangle103}>
          <input
            type="text"
            placeholder='이메일 찾기'
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={style.inputField103} />
        </div>
      </form>
    </>
  )
}

export default FindAccountPage;
