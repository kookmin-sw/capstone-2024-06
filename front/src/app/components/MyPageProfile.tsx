import Image from "next/image";

const MyPageProfile = () => {
  return (
    <main className="w-full mt-10 border h-[500px] flex justify-center items-center">
      <div className="flex flex-col justify-center items-center">
        <div className="flex justify-center items-center border w-[500px] h-[200px]">
          <div className="h-[100px] w-[100px]">
            <Image
              src="/Profilex2.webp"
              alt="Profile image"
              width={1000}
              height={1000}
              objectFit="cover"
              className="cursor-pointer mr-1 rounded-full"
            />
          </div>
          <div className="flex-col">
            <div className="text-2xl">박근우</div>
          </div>
        </div>
        <div className="flex justify-center items-center w-full border h-10">
          <div className="w-[300px] border text-center bg-[#ced4da] text-[#f8f9fa]">
            프로필 수정
          </div>
        </div>
        <div className="flex justify-center">
          <div>팔로잉</div>
          <div>0</div>
          <div>팔로워</div>
          <div>0</div>
        </div>
      </div>
    </main>
  );
};

export default MyPageProfile;
