"use client";
import { SetStateAction, useState, ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";


interface ImagePreview {
  url: string;
  file: File;
}

const PostCreates = () => {
  const router = useRouter();

  const [PostCreateTitle, SetPostCreateTitle] = useState("");

  const PostCreateTitleChange = (e: {
    target: { value: SetStateAction<string> };
  }) => {
    SetPostCreateTitle(e.target.value);
  };

  const [PostCreateContent, SetPostCreateContent] = useState("");

  const PostCreateContentChange = (e: {
    target: { value: SetStateAction<string> };
  }) => {
    SetPostCreateContent(e.target.value);
  };

  const {data: session} = useSession()
  
 
  const PostCreateBt = async () => {
    try {
      const PostCreateData = {
        title: PostCreateTitle,
        category: "None",
        content: PostCreateContent,
      };
      const response = await fetch(`${process.env.Localhost}/post`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${session?.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(PostCreateData),
      });
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Error", error);
    }
    router.push("/Community");
  };
  
  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null); 

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImagePreview({ url, file });
    }
  };
  const TestBt = async () => {
    try {
      const formData = new FormData();
      formData.append("file",  imagePreview.file);

      const response = await fetch(`http://192.168.194.28:8080/process_image/`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log(data.result_filename);
      const test = await fetch(`http://192.168.194.28:8080/get_image/${data.result_filename}`,{
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })
      const tests = await test;
      console.log(tests.url)

    } catch (error) {
      console.error("Error", error);
    }
  };

  return (
    <main>
      <input type="file" onChange={handleFileChange} />
      <div className="flex mt-10 border-b pb-2 text-[#808080] text-3xl font-semi">
        <input
          type="text"
          placeholder="제목을 입력해주세요 !"
          className="border-none focus:outline-none text-3xl font-semi w-[95%]"
          value={PostCreateTitle}
          onChange={PostCreateTitleChange}
        />

        <div>{PostCreateTitle.length}/20</div>
      </div>
      <div className="flex-wrap mt-10 pb-2 text-[#808080] text-3xl font-semi">
        <textarea
          placeholder="내용을 입력해주세요"
          className="border-none focus:outline-none h-96 text-3xl font-semi w-[95%] resize-none"
          value={PostCreateContent}
          onChange={PostCreateContentChange}
        />
      </div>
      <button
        className="bg-transparent hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
        onClick={PostCreateBt}
      >
        글 작성하기
      </button>
      <button className="" onClick={TestBt}>
        테스트 버튼
      </button>
    </main>
  );
};

export default PostCreates;
