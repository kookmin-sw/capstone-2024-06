"use client";
import { signIn } from "next-auth/react";
import { ChangeEvent, FormEvent, useState } from "react";
import style from "./signinStyle.module.css";


type LoginInput = {
  username: string;
  password: string;
}

type PageProps = {
  searchParams: { error?: string }
}

export default function LoginPage({ searchParams }: PageProps) {
  const [inputs, setInputs] = useState<LoginInput>({ username: "", password: "" });

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const name = event.target.name;
    const value = event.target.value;
    setInputs(values => ({ ...values, [name]: value }))
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await signIn("credentials", {
      username: inputs.username,
      password: inputs.password,
      callbackUrl: '/'
    });
  }

  const handlekakao = async (event: FormEvent) => {
    event.preventDefault();
    await signIn("kakao", {
      redirect: true,
      callbackUrl: '/'
    });
  }

  const handlenaver = async (event: FormEvent) => {
    event.preventDefault();
    await signIn("naver", {
      redirect: true,
      callbackUrl: '/'
    });
  }

  const handlegoogle = async (event: FormEvent) => {
    event.preventDefault();
    await signIn("google", {
      redirect: true,
      callbackUrl: '/'
    });
  }


  return (
    <>
      <div className={style.title}>what_desk</div>
      <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8">
        <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="username" className="block text-sm font-medium leading-6 text-gray-900">
                아이디
              </label>
              <div className="mt-">
                <input
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="off"
                  required
                  value={inputs.username || ""}
                  onChange={handleChange}
                  className="block w-full rounded-md border-0 px-2 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-yellow-600 sm:text-sm sm:leading-6"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="block text-sm font-medium leading-6 text-gray-900">
                  비밀번호
                </label>
              </div>
              <div className="mt-2">
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="off"
                  required
                  value={inputs.password || ""}
                  onChange={handleChange}
                  className="block w-full rounded-md border-0 px-2 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-yellow-600 sm:text-sm sm:leading-6"
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                className="flex w-full justify-center rounded-md bg-yellow-700 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                로그인
              </button>
            </div>
            {searchParams.error && (
              <p className="text-red-600 text-center capitalize">
                Login failed.
              </p>
            )}
          </form>
          <div>
            간편 로그인
          </div>
          <form className="space-y-6" onSubmit={handlekakao}>
            <button
              onClick={() => signIn("kakao", { redirect: true, callbackUrl: "/" })}
              className="flex w-full justify-center rounded-md bg-yellow-300 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              <img src='https://developers.kakao.com/tool/resource/static/img/button/kakaotalksharing/kakaotalk_sharing_btn_small.png'
                alt="Kakao Logo"
                className="w-6 h-6" />
              카카오 로그인
            </button>
          </form>
          <line className="flex items-center justify-center mt-2"></line>
          <form className="space-y-6" onSubmit={handlenaver}>
            <button
              onClick={() => signIn("naver", { redirect: true, callbackUrl: "/" })}
              className="flex w-full justify-center rounded-md bg-green-500 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              <img src='https://logoproject.naver.com/favicon.ico'
                alt="naver Logo"
                className="w-6 h-6" />
              네이버 로그인
            </button>
          </form>
          <line className="flex items-center justify-center mt-2"></line>
          <form className="space-y-6" onSubmit={handlegoogle}>
            <button
              onClick={() => signIn("google", { redirect: true, callbackUrl: "/" })}
              className="flex w-full justify-center rounded-md bg-slate-50 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              <img src='https://w7.pngwing.com/pngs/326/85/png-transparent-google-logo-google-text-trademark-logo-thumbnail.png'
                alt="Google Logo"
                className="w-6 h-6" />
              구글 로그인
            </button>
          </form>
        </div>
      </div>
    </>
  )
}