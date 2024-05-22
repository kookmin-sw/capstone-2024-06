"use client";
import { signIn } from "next-auth/react";
import { ChangeEvent, FormEvent, useState } from "react";
import Link from "next/link";
import Nav from "../../../components/Nav";


type LoginInput = {
  user_id: string;
  password: string;
}

type PageProps = {
  searchParams: { error?: string }
}

export default function LoginPage({ searchParams }: PageProps) {
  const [inputs, setInputs] = useState<LoginInput>({ user_id: "", password: "" });

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const name = event.target.name;
    const value = event.target.value;
    setInputs(values => ({ ...values, [name]: value }))
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await signIn("credentials", {
      user_id: inputs.user_id,
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
      <Nav />
      <div className="flex justify-center text-[#A26D07] text-5xl font-normal mt-24">What Desk</div>
      <line className="flex items-center justify-center mt-10"></line>
      <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-sm">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="user_id" className="block text-sm font-medium leading-6 text-gray-800">
                아이디
              </label>
              <div className="mt-1">
                <input
                  id="user_id"
                  name="user_id"
                  type="text"
                  autoComplete="off"
                  required
                  value={inputs.user_id || ""}
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
              <div className="mt-1">
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
                className="flex w-full justify-center rounded-md bg-yellow-700 px-3 py-1.5 text-md font-semibold leading-6 text-white shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
              >
                로그인
              </button>
            </div>
            <div className="flex mt-2">
              <div className="mr-auto">
                <Link href="/login/sign-up">
                  <span className="text-gray-500 text-sm hover:underline">회원가입</span>
                </Link>
              </div>
              <div className="ml-auto">
                <Link href="/findAccount">
                  <span className="text-gray-500 text-sm hover:underline">계정 찾기</span>
                </Link>
              </div>
            </div>


            {searchParams.error && (
              <p className="text-red-600 text-center capitalize">
                Login failed.
              </p>
            )}
          </form>
          <line className="flex items-center justify-center mt-5"></line>
          <svg className="mt-2" width="100%" height="2">
            <line x1="0" y1="0" x2="100%" y2="0" stroke="hsl(39, 70%, 70%)" strokeWidth="3" />
          </svg>
          <line className="flex items-center justify-center mt-5"></line>
          <form className="space-y-6" onSubmit={handlekakao}>
            <button
              onClick={() => signIn("kakao", { redirect: true, callbackUrl: "/" })}
              className="flex w-full justify-center rounded-md bg-yellow-300 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
            >
              <img src='https://developers.kakao.com/tool/resource/static/img/button/kakaotalksharing/kakaotalk_sharing_btn_small.png'
                alt="Kakao Logo"
                className="w-6 h-6 mr-2" />
              카카오 로그인
            </button>
          </form>
          <line className="flex items-center justify-center mt-2"></line>
          <form className="space-y-6" onSubmit={handlenaver}>
            <button
              onClick={() => signIn("naver", { redirect: true, callbackUrl: "/" })}
              className="flex w-full justify-center rounded-md bg-green-500 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
            >
              <img src='https://logoproject.naver.com/favicon.ico'
                alt="naver Logo"
                className="w-6 h-6 mr-2" />
              네이버 로그인
            </button>
          </form>
          <line className="flex items-center justify-center mt-2"></line>
          <form className="space-y-6" onSubmit={handlegoogle}>
            <button
              onClick={() => signIn("google", { redirect: true, callbackUrl: "/" })}
              className="flex w-full justify-center rounded-md bg-slate-50 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
            >
              <img src='https://w7.pngwing.com/pngs/326/85/png-transparent-google-logo-google-text-trademark-logo-thumbnail.png'
                alt="Google Logo"
                className="w-6 h-6 mr-2" />
              구글 로그인
            </button>
          </form>
        </div>
      </div>
    </>
  )
}