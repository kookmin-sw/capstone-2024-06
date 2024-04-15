// "use client";
// import { useEffect, useState } from 'react';
// import { useRouter } from 'next/router';
// import { useSession } from 'next-auth/react';

// const Follower = () => {
//   const { data: session } = useSession();
//   const [followers, setFollowers] = useState([]);
//   const [followerCount, setFollowerCount] = useState(0);

//   useEffect(() => {
//     const fetchFollowings = async () => {
//       try {

//         // 팔로잉 정보를 가져오는 API 요청
//         const res = await fetch(`${process.env.Localhost}/followers`, {
//           method: 'POST',
//           headers: {
//             'Content-Type': 'application/json',
//             Authorization: `Bearer ${(session as any)?.access_token}`,
//           },
//         });

//         // 요청 성공 시 데이터 설정
//         if (res.ok) {
//           const data = await res.json();
//           console.log(data);
//           setFollowers(data);
//           setFollowerCount(data);
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
//         팔로워 {followerCount}
//       </div>
//       {followers && followers.map((follower) => (
//         <div key={follower.user_id}>
//           <h2>{follower.username}</h2>
//           <img src={follower.image} alt={follower.username} />
//         </div>
//       ))}
//     </div>
//   );
// };

// export default Follower;