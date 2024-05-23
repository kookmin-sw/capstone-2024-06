"use client";
import { useState, useEffect } from 'react';
import Link from 'next/link';
import Nav from '../../../components/Nav';

const FindUserIDpage = () => {
  const [email, setEmail] = useState<string>('');
  const [foundUserId, setFoundUserId] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  useEffect(() => {
    const fetchUserId = async () => {
      if (!isSubmitting) return;

      try {
        const response = await fetch(`/api/user/find/id/${email}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'Something went wrong');
        }

        console.log(response);
        const data = await response.json();

        setFoundUserId(data.user_id);
        setError('');
      } catch (error : any) {
        setError(error.message);
        setFoundUserId('');
      } finally {
        setIsSubmitting(false);
      }
    };

    fetchUserId();
  }, [isSubmitting, email]);

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitting(true);
  };

  return (
    <>
      <Nav />
      <form onSubmit={onSubmit} className="flex flex-col items-center justify-center w-full h-full p-20">
        <div className="text-4xl font-bold text-yellow-700 mb-8">what_desk</div>
        <div className="text-lg font-semibold text-gray-700 mb-4">아이디 찾기</div>
        <div className="w-full max-w-md bg-white p-6 rounded-lg shadow-md">
          <div className="text-sm font-semibold text-gray-700 mb-4">등록된 이메일을 입력해주세요</div>
          <label className="block text-gray-700 font-bold mb-2" htmlFor="email">이메일</label>
          <input
            id="email"
            type="email"
            placeholder='email'
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded mb-4"
          />
          {error && (
            <div className="text-red-500 mb-4">{error}</div>
          )}
          {foundUserId && (
            <div className="text-green-500 mb-4">등록된 아이디: {foundUserId}</div>
          )}
          <button type="submit" className="w-full bg-yellow-500 text-white font-bold py-2 rounded hover:bg-yellow-600">
            확인
          </button>
        </div>
        <Link href="/login/sign-in" className="text-yellow-900 mt-4">로그인</Link>
      </form>
    </>
  );
};

export default FindUserIDpage;
