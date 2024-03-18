"use client";
import { useState } from 'react';
import Link from 'next/link';
import Nav from '../components/Nav';
import style from './findAccountStyle.module.css';

const FindAccountPage = () => {

  return (
    <>
      <Nav />
      <form className={style.desktop}>
        <div className={style.title}>what_desk</div>
        <div className={style.rectangle50}></div>
        <div className={style.rectangle51}>아이디 찾기</div>
      </form>
    </>
  )
}

export default FindAccountPage;