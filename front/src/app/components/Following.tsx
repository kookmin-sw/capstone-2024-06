// "use client";
// import { useEffect, useState } from 'react';
// import { useRouter } from 'next/router';
// import { useSession } from 'next-auth/react';

// const Following = () => {
//   const { data: session } = useSession();
//   const [followings, setFollowings] = useState([]);
//   const [followingCount, setFollowingCount] = useState(0);

//   useEffect(() => {
//     const fetchFollowings = async () => {
//       try {


//         // 팔로잉 정보를 가져오는 API 요청
//         const res = await fetch(`${process.env.Localhost}/followees`, {
//           method: 'POST',
//           headers: {
//             'Content-Type': 'application/json',
//             Authorization: `Bearer ${(session as any)?.access_token}`,
//           },
//         });

//         // 요청 성공 시 데이터 설정
//         if (res.ok) {
//           const data = await res.json();
//           setFollowings(data);
//           setFollowingCount(data);
//         } else {
//           console.error('Failed to fetch followings');
//         }
//       } catch (error) {
//         console.error(error);
//       }
//     };

//     fetchFollowings();
//   }, [session]);



//   return (
//     <div>
//       <div className="absolute w-[173px] h-[83px] left-[697px] top-[99px] font-semibold text-base leading-9 text-black hover:text-[#F4A460]"
//       >
//         팔로잉 {followingCount}
//       </div>
//       {followings.map((following) => (
//         <div key={following.user_id}>
//           <h2>{following.username}</h2>
//           <img src={following.image} alt={following.username} />
//         </div>
//       ))}
//     </div>
//   );
// };

// export default Following;