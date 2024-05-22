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
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex  justify-center w-full h-auto">
        <div className="flex-col items-center justify-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <form onSubmit={onSubmit} className="">
            <div className="flex justify-center text-[#A26D07] text-5xl font-normal mt-12">
              What Desk
            </div>

            <div className="flex justify-center w-full mt-20">
              <div className="w-[400px] mb-6">
                <input
                  type="text"
                  className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                  placeholder="닉네임"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            </div>
            <div className="flex justify-center w-full mt-5">
              <div className="w-[400px] mb-6">
                <input
                  type="text"
                  className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                  placeholder="아이디"
                  value={user_id}
                  onChange={(e) => setUserId(e.target.value)}
                />
              </div>
            </div>

            <div className="flex justify-center w-full">
              <div className="w-[400px]">
                <input
                  type="text"
                  className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                  placeholder="비밀번호"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>
            <div className="flex justify-center w-full mt-1">
              <div className="w-[400px] mb-6">
                <input
                  type="text"
                  className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                  placeholder="비밀번호 확인"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
            </div>

            {error && <div className={style.errormessage59}>{error}</div>}
            <div className="flex justify-center w-full mt-5">
              <div className="w-[400px] mb-6">
                <input
                  type="text"
                  className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                  placeholder="이메일"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>
            <div className="flex justify-center">
              <div className="w-[400px] border border-b-[#a69067]"></div>
            </div>
            <div className="flex justify-center mt-5">
             
                <button
                  type="submit"
                  className="w-[400px] bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                >
                  회원가입
                </button>
              
            </div>
            <div className="flex mt-3 w-full justify-center">
              <div className="mr-10">
                <Link href="/login/sign-in">
                  <div>로그인</div>
                </Link>
              </div>
              <div className="">
                <Link href="/findAccount">
                  <div>계정찾기</div>
                </Link>
              </div>
            </div>
          </form>
        </div>
      </div>
    </main>
  );
};

export default RegistrationForm;