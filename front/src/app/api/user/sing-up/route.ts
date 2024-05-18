import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

const RegistrationForm = () => {
  const [user_id, setUserId] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [image, setImage] = useState('');
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
          router.push('/login'); // redirect to login page or any other page
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

};

export default RegistrationForm;