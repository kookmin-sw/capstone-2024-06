
import { useRouter } from 'next/router';
import Image from "next/image";
import { routeModule } from 'next/dist/build/templates/app-page';

const handleSubmit = async (
  user_id: string,
  name: string,
  email: string,
  image: string,
  password: string,
  confirmPassword: string,
  setError: (error: string) => void,
) => {

  // 비밀번호 조건 확인
  const passwordRegex = /^(?=.*[0-9])(?=.*[!@#$%^&*])[a-zA-Z0-9!@#$%^&*]{8,16}$/;
  if (!passwordRegex.test(password)) {
    setError(
      '비밀번호는 8~16자의 영문, 숫자, 특수문자(!,@,#,$,%,^,&,*)를 포함해야 합니다.'
    );
    return;
  }

  // 비밀번호 일치 확인
  if (password !== confirmPassword) {
    setError('비밀번호가 일치하지 않습니다.');
    return;
  }

  // 회원가입 요청
  try {
    const response = await fetch("http://175.194.198.155:8080/user", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_id, name, email, image, password }),
    });
    const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
    };

    if (response.ok) {
      // 회원가입 성공
      console.log('회원가입이 완료되었습니다.');
    } else {
      // 회원가입 실패
      setError('회원가입에 실패했습니다.');
    }
  } catch (error) {
    console.error('회원가입 중 오류 발생:', error);
    setError('회원가입 중 오류가 발생했습니다.');
  }

};

export default handleSubmit;