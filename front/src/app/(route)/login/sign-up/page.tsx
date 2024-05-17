"use client";
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import style from './signupStyle.module.css';
import Nav from '../../../components/Nav';

const RegistrationForm = () => {
  const [user_id, setUserId] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const image = "/Profilex2.webp";
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

  useEffect(() => {
    if (!isSubmitting) return;

    const handleSubmit = async () => {
      // 비밀번호 조건 확인
      const passwordRegex = /^(?=.*[0-9])(?=.*[!@#$%^&*])[a-zA-Z0-9!@#$%^&*]{8,16}$/;
      if (!passwordRegex.test(password)) {
        setError(
          '비밀번호는 8~16자의 영문, 숫자, 특수문자(!,@,#,$,%,^,&,*)를 포함해야 합니다.'
        );
        setIsSubmitting(false);
        return;
      }

      // 비밀번호 일치 확인
      if (password !== confirmPassword) {
        setError('비밀번호가 일치하지 않습니다.');
        setIsSubmitting(false);
        return;
      }

      // 회원가입 요청
      try {
        const response = await fetch(`${process.env.Localhost}/user`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ user_id, name, email, image, password }),
        });

        if (response.ok) {
          // 회원가입 성공
          console.log('회원가입이 완료되었습니다.');
          router.push('/login/sign-in'); // redirect to login page or any other page
        } else {
          // 회원가입 실패
          setError('회원가입에 실패했습니다.');
        }
      } catch (error) {
        console.error('회원가입 중 오류 발생:', error);
        setError('회원가입 중 오류가 발생했습니다.');
      } finally {
        setIsSubmitting(false);
      }
    };

    handleSubmit();
  }, [isSubmitting, user_id, name, email, image, password, confirmPassword, router]);

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitting(true);
  };

  // const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
  //   e.preventDefault();
  //   await handleSubmit(user_id, name, email, image, password, confirmPassword, setError);
  // };

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
            onChange={(e) => setName(e.target.value)}
            className={style.inputField63} />
        </div>
        <div className={style.rectangle64}>
          <input
            type="text"
            placeholder='아이디'
            value={user_id}
            onChange={(e) => setUserId(e.target.value)}
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
          <Link href="/login/sign-in">
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

export default RegistrationForm;