import { useState } from 'react';
import handleSubmit from '../api/user/sign-up/route';
import style from './signupStyle.module.css';

const RegisterForm = () => {
  const [user_id, setUserid] = useState<string>('');
  const [name, setUsername] = useState<string>('');
  const [email, setEmail] = useState<string>('');
  const [image, setImage] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [error, setError] = useState<string>('');

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    await handleSubmit(user_id, name, email, image, password, confirmPassword, setError);
  };



  return (
    <form onSubmit={onSubmit} className={style.desktop}>
      <div className={style.rectangle64}>
        <input type="text" placeholder="아이디"
          value={user_id} onChange={(e) => setUserid(e.target.value)} />
      </div>
      <div className={style.rectangle58}>
        <input type="password" placeholder="비밀번호"
          value={password} onChange={(e) => setPassword(e.target.value)} />
      </div>
      <div className={style.rectangle59}>
        <input type="password" placeholder="비밀번호 확인"
          value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} />
      </div>
      <div className={style.rectangle63}>
        <input type="text" placeholder='닉네임'
          value={name} onChange={(e) => setUsername(e.target.value)} />
      </div>
      <div className={style.rectangle62}>
        <input type="email" placeholder='이메일'
          value={email} onChange={(e) => setEmail(e.target.value)} />
      </div>
      {error && <div>{error}</div>}
    </form>
  );
};

export default RegisterForm;
