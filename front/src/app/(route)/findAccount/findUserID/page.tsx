/*DESKTOP22*/
"use client"
import { useRouter } from 'next/navigation'

import { useState } from 'react';
import Link from 'next/link';
import handleSubmit from '../../../api/user/findUserID/route';
import style from './findUserIDStyle.module.css';
import Nav from '../../../components/Nav';

const FindUserIDpage = () => {
  const [nickname, setNickname] = useState<string>('');
  const [email, setEmail] = useState<string>('');
  const [foundUserId, setFoundUserId] = useState<string>('');
  const [error, setError] = useState<string>('');

  const router = useRouter();
  const handlePasswordButtonClick = () => {
    router.push('/findAccountID');
    // 비밀번호 찾기 버튼을 클릭했을 때의 동작을 여기에 작성합니다.
    // 예: 비밀번호 찾기 모달을 열거나 다른 페이지로 이동합니다.
  };

  const handleEmailButtonClick = () => {
    console.log('이메일 찾기 버튼이 클릭되었습니다.');
    router.push('/findUserID');
    // 여기에 이메일 찾기와 관련된 추가적인 로직을 작성하세요.
  };


  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const data = await handleSubmit(email, setFoundUserId, setError);
      setFoundUserId(data.user_id);
      setError('');
    } catch (error) {
      setError(error.message);
      setFoundUserId('');
    }
  };

  return (
    <>
      <Nav />
      <form onSubmit={onSubmit} className={style.desktop}>
      <div>
      <div className={style.Desktop}></div>
          <button className={style.findID} onClick={handlePasswordButtonClick}>비밀번호 찾기</button>
        </div>
        <div>
          <button className={style.findPW} onClick={handleEmailButtonClick}>아이디 찾기</button>
        </div>
        
        
        <div className={style.logo}>어떤대스크</div>
        <div className={style.line13}></div>
        <div className={style.registeredEmail}>닉네임과 등록된 이메일을 입력해주세요</div>
        <div className={style.nickname}>닉네임</div>
        <div className={style.emailAddress}>이메일주소</div>
        <div className={style.Rectangle103}></div>
        <div className={style.Rectangle105}>
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

export default FindUserIDpage;
