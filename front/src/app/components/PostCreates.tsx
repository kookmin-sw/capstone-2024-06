"use client";
import { SetStateAction, useState, ChangeEvent, DragEvent } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import SelectCategory from "./SelectCategory"

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

  const { data: session } = useSession();

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

  const [SelectedCategory, SetSelectedCategory] = useState("자유");

  const SelectCategoryChange = (category: string) => {
    SetSelectedCategory(category);
    console.log(SelectedCategory);
  };
  // const TestBt = async () => {
  //   try {
  //     const formData = new FormData();
  //     formData.append("file",  imagePreview.file);

  //     const response = await fetch(`http://192.168.194.28:8080/process_image/`, {
  //       method: "POST",
  //       body: formData,
  //     });
  //     const data = await response.json();
  //     console.log(data.result_filename);
  //     const test = await fetch(`http://192.168.194.28:8080/get_image/${data.result_filename}`,{
  //       method: "GET",
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //     })
  //     const tests = await test;
  //     console.log(tests.url)

  //   } catch (error) {
  //     console.error("Error", error);
  //   }
  // };

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
      <SelectCategory OnSelectCategory={SelectCategoryChange}/>
      <button
        className="bg-transparent hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded"
        onClick={PostCreateBt}
      >
        글 작성하기
      </button>
    </main>
  );
};

export default PostCreates;
