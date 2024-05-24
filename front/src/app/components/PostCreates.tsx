"use client";
import {
  SetStateAction,
  useState,
  ChangeEvent,
  DragEvent,
  useEffect,
  useReducer,
} from "react";
import { useRouter, useParams } from "next/navigation";
import { useSession } from "next-auth/react";

interface ImagePreview {
  url: string;
  file: File;
}

const PostCreates = () => {
  const { data: session } = useSession();

  const router = useRouter();

  // Temp ID GET
  const PostCreateTempId = useParams();
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

  const PostCreateBt = async () => {
    try {
      // image Post
      if (imagePreview) {
        const formData = new FormData();
        formData.append("file", imagePreview.file);
        const ImagePost = await fetch(
          `/what-desk-api/community/image/${PostCreateTempId.PostCreateId}`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
            body: formData,
          }
        );
        const ImageDatas = await ImagePost.json();
        console.log(ImageDatas);
      }
      // Posting Post
      const PostCreateData = {
        title: PostCreateTitle,
        category: Category,
        content: PostCreateContent,
      };
      console.log(PostCreateData)
      const PostCreate = await fetch(
        `/what-desk-api/community/post/${PostCreateTempId.PostCreateId}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${(session as any)?.access_token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(PostCreateData),
        }
      );
      const PostCreateDatas = await PostCreate.json();
      console.log(PostCreateDatas);
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

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      const url = URL.createObjectURL(file);
      setImagePreview({ url, file });
    }
  };

  const ImageDeleteBt = () => {
    setImagePreview(null);
  };

  const reducer = (state: any, action: any) => {
    switch (action.type) {
      case "자유":
        return { Category: "자유" };
      case "인기":
        return { Category: "인기" };
      case "삽니다":
        return { Category: "삽니다" };
      case "팝니다":
        return { Category: "팝니다" };
      default:
        return state;
    }
  };

  const [Category, SetCategory] = useState('자유');
  const handleChange = (e : any) => {
    SetCategory(e.target.value);
  };

  return (
    <main>
      <div className="flex flex-col items-center justify-center mt-4">
        <div
          className="m-2 relative"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {imagePreview ? (
            <img
              src={imagePreview.url}
              alt="Image preview"
              className="w-[500px] h-[400px] cursor-pointer"
            />
          ) : (
            <div className="flex my-4 items-center justify-center border-dashed border-2 text-[#808080] text-sm w-[600px] h-[400px] cursor-pointer">
              드래그하여 사진 업로드
            </div>
          )}
        </div>
        {!imagePreview ? (
          <div className="flex">
            <input
              type="file"
              id="file-upload"
              className="hidden"
              onChange={handleFileChange}
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
            >
              파일 선택
            </label>
          </div>
        ) : (
          <div className="flex">
            <div
              className="cursor-pointer bg-blue-500 text-white flex items-center justify-center w-[100px] h-[45px] rounded hover:scale-105"
              onClick={ImageDeleteBt}
            >
              사진제거
            </div>
          </div>
        )}
      </div>
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
      <div className="">
        <div className="flex justify-center mb-2 text-2xl text-gray-600 dark:text-white">
          카테고리
        </div>
        <div className="flex space-x-2 justify-center">
          <select
            value={Category} 
            onChange={handleChange} 
            className="w-[100px] bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
          >
            <option value="자유">자유</option>
            <option value="팝니다">팝니다</option>
            <option value="삽니다">삽니다</option>
            <option value="인기">인기</option>
            <option value="실시간">실시간</option>
          </select>
        </div>
      </div>

      <div className="flex w-full justify-center">
        <button
          className="my-10 bg-transparent hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
          onClick={PostCreateBt}
        >
          글 작성하기
        </button>
      </div>
    </main>
  );
};

export default PostCreates;
