import { ChangeEventHandler, FormEventHandler, useState } from 'react';
import { useSession } from 'next-auth/react';

const EditUserInfo = () => {
  const { data: session, update } = useSession();
  const [userInfo, setUserInfo] = useState({
    name: session?.user?.name || '',
    email: session?.user?.email || '',
    newPassword: '',
    confirmPassword: '',
  });
  const [message, setMessage] = useState('');

  const handleChange: ChangeEventHandler<HTMLInputElement> = (e) => {
    const { name, value } = e.target;
    setUserInfo((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit: FormEventHandler<HTMLFormElement> = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${process.env.Localhost}/user/modification`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${(session as any)?.access_token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userInfo),
      });
      if (response.ok && session != null) {
        session.user = await response.json();
        update(session);
        setMessage('User information updated successfully.');
      } else {
        const data = await response.json();
        setMessage(data.error || 'Failed to update user information.');
      }
    } catch (error) {
      console.error('Error updating user information:', error);
      setMessage('An error occurred while updating user information.');
    }
  };

  return (
    <div >
      <div className="absolute w-[173px] h-[83px] left-[700px] top-[270px] font-semibold text-base leading-9 text-black">
        회원 정보 수정
      </div>
      <form onSubmit={handleSubmit}>
        <div className='absolute left-[700px] top-[350px] block text-sm font-medium leading-6 text-gray-900'>
          <label htmlFor="name">닉네임</label>
          <input
            type="text"
            id="name"
            name="name"
            value={userInfo.name}
            onChange={handleChange}
            className="block w-full rounded-md border-0 px-2 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-yellow-600 sm:text-sm sm:leading-6"
          />
        </div>
        <div className='absolute left-[700px] top-[420px] block text-sm font-medium leading-6 text-gray-900'>
          <label htmlFor="email">이메일</label>
          <input
            type="email"
            id="email"
            name="email"
            value={userInfo.email}
            onChange={handleChange}
            className="block w-full rounded-md border-0 px-2 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-yellow-600 sm:text-sm sm:leading-6"
          />
        </div>
        <div className='absolute left-[700px] top-[490px] block text-sm font-medium leading-6 text-gray-900'>
          <label htmlFor="newPassword">변경할 비밀번호</label>
          <input
            type="password"
            id="newPassword"
            name="newPassword"
            value={userInfo.newPassword}
            onChange={handleChange}
            className="block w-full rounded-md border-0 px-2 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-yellow-600 sm:text-sm sm:leading-6"
          />
        </div>
        <div className='absolute left-[700px] top-[560px] block text-sm font-medium leading-6 text-gray-900'>
          <label htmlFor="confirmPassword">비밀번호 확인</label>
          <input
            type="password"
            id="confirmPassword"
            name="confirmPassword"
            value={userInfo.confirmPassword}
            onChange={handleChange}
            className="block w-full rounded-md border-0 px-2 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-yellow-600 sm:text-sm sm:leading-6"

          />
        </div>
        <button className='absolute left-[700px] top-[690px] justify-center rounded-md bg-yellow-700 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500'
          type="submit">회원 정보 저장</button>
      </form>
    </div>
  );
};

export default EditUserInfo;
