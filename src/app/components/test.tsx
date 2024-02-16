"use client";

const test = () => {
  const handleClick = async () => {
    try {
      const formData = new URLSearchParams();
      formData.append("grant_type", "password");
      formData.append("username", "username1");
      formData.append("password", "password");
      const response = await fetch(
        "http://172.16.101.247:8080/user/oauth_sign_in",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: formData.toString(),
        }
      );

      if (response.ok) {
        console.log("POST 요청이 성공했습니다.");
        // 성공적으로 처리된 경우 추가적인 작업 수행
      } else {
        console.error("POST 요청이 실패했습니다.");
        // 요청이 실패한 경우 에러 처리
      }
    } catch (error) {
      console.error("POST 요청 중 에러가 발생했습니다.", error);
      // 요청 중 에러 발생 시 처리
    }
  };

  const handleClicks = async () => {
    try {
      const formData = new URLSearchParams();
      formData.append("grant_type", "password");
      formData.append("username", "username1");
      formData.append("password", "password");
      const response = await fetch("http://192.168.93.253:8080/user/me", {
        method: "GET",
        headers: {
          Authorization:
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VybmFtZTEiLCJleHAiOjE3MDgwNjk1MzZ9.vGshZDi2D40T-55sQKbeHumHOUOVGw1bm06cFpeqbwM",
        },
      });

      if (response.ok) {
        console.log("POST 요청이 성공했습니다.");
        // 성공적으로 처리된 경우 추가적인 작업 수행
      } else {
        console.error("POST 요청이 실패했습니다.");
        // 요청이 실패한 경우 에러 처리
      }
    } catch (error) {
      console.error("POST 요청 중 에러가 발생했습니다.", error);
      // 요청 중 에러 발생 시 처리
    }
  };
  return (
    <div>
      <button onClick={handleClick}>로그인</button>
      <button onClick={handleClicks}>me</button>;
    </div>
  );
};

export default test;
