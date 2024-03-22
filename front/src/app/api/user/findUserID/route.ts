import { useRouter } from 'next/router';

const handleSubmit = async (
  email: string,
  setFoundUserId: (userId: string) => void,
  setError: (error: string) => void
) => {
  // 조건확인
  const emailRegex = /@/;
  if (!emailRegex.test(email)) {
    setError(
      '올바른 형식이 아닙니다. 다시 입력해주세요.'
    );
    return;
  }
  // 계정 찾기 요청
  try {
    const response = await fetch("http://10.223.114.14:8080/user", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email }),
    });
    const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
    };

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

export default handleSubmit;