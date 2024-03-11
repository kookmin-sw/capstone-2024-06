"use client";

const test = () => {
  const handleClick = async () => {
    try {
      const formDatas = {
        post_id: "123",
        title: "Test 1",
        content: "테스트입니다",
        category: "자유게시판",
        writer_username: "user1"
      };

      const formData = new URLSearchParams();
      formData.append("post_id", "2");
      formData.append("title", "Test 1");
      formData.append("content", "테스트입니다");
      formData.append("category", "자유게시판");
      formData.append("writer_username", "User 1");
      const keyword = "테스트입니다";
      const encodedKeyword = encodeURIComponent(keyword);
      const response = await fetch(
        `http://175.194.198.155:8080/post/search?keyword=테스트입니다`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
          // body: JSON.stringify(formDatas),
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
      <button className="border" onClick={handleClick}>Test</button>
      <button onClick={handleClicks}>me</button>;
      {/* <div className="flex-col items-center border min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <ImageSlider />

          <div className="font-bold text-lg">잘꾸민 책상 모음</div>
          <RecommendImgSlider />
        </div> */}
    </div>
  );
};

export default test;

