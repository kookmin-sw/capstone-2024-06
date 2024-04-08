"use client"
import { useState } from 'react';
import Link from 'next/link';
import handleSubmit from '../api/user/findUserID/route';
import style from './findUserIDresultStyle.module.css';
import Nav from '../components/Nav';

const findUserIDresult = () => {


  return (
    <>
    <Nav/>
    <div>
    <div className={style.Desktop}></div>
    <div className={style.topbar}></div>
    <div className={style.Rectangle103}></div>
    <div className={style.logo}>어떤대스크</div>
    <div className={style.notice}>회원님의 아이디는 yujin****입니다.</div>
    <div className={style.line13}></div>
    <div className={style.findPW}>아이디 찾기</div>
    <div className={style.findID}>비밀번호 찾기</div>

  </div>
  </>
  )
};

export default findUserIDresult;
