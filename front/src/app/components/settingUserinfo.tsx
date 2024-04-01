// SettingForm.tsx
"use client";
import { useState, FormEvent } from 'react'
import { useSession } from 'next-auth/react'

interface IProps {
  user: {
    name: string;
    user_id: string;
  } | null;
}

export default function SettingForm({ user }: IProps) {
  const [name, setName] = useState(user?.name ?? '');
  const [blankName, setBlankName] = useState(false)
  const [change, setChange] = useState(false);
  const { data: session, status, update } = useSession()


  const onChangeName = (e: React.ChangeEvent<HTMLInputElement>) => {
    setChange(true)
    setName(e.target.value)
  }

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!blankName && user) { // user가 null이 아닌 경우에만 실행
      fetch(`${process.env.api_url}/user`, {
        method: 'POST',
        body: JSON.stringify({ id: user.user_id, name }),
      }).then((res) => {
        if (res.status === 200) {
          if (status === "authenticated") update({ name })
          window.alert('변경되었습니다.')
        }
      })
    }
  }

  return (
    <>
      <div className="absolute left-[750px] top-[400px]">
        <form onSubmit={onSubmit} className="">
          <input
            name="id"
            type="text"
            value={name}
            onChange={onChangeName}
            onBlur={() => setBlankName(name.length <= 0)}
            placeholder="닉네임"
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:border-blue-500 mb-4"
          />
          <button
            type="submit"
            disabled={!change}
            className="absolute left-[110px] top-[220px] bg-yellow-600 px-3 py-1.5 text-sm font-semibold leading-6 text-white rounded-md shadow-sm hover:bg-yellow-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
          >
            저장
          </button>
        </form>
      </div>
    </>
  );

}
