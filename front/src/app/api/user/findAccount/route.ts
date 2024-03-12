import { useState } from 'react';

const FindAccount = () => {
  const [email, setEmail] = useState<string>('');
  const [foundUserId, setFoundUserId] = useState<string>('');
  const [error, setError] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const formData = {
        email
      };

      const response = await fetch('/api/findAccount', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        const data = await response.json();
        setFoundUserId(data.user_id); // 이메일이 존재하면 해당 사용자의 ID를 설정
        setError(''); // 에러 메시지 초기화
      } else {
        setError('회원정보가 없습니다'); // 에러 메시지 설정
      }
    } catch (error) {
      console.error('Error finding user:', error);
      setError('다시 시도해주세요'); // 서버 오류 시 에러 메시지 설정
    }
  };
};

export default FindAccount;
