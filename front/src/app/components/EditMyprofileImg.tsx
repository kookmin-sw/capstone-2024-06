import { useState, ChangeEvent, FormEvent, useEffect, use } from 'react';
import { useSession } from 'next-auth/react';

interface ImagePreview {
  url: string;
  file: File;
}

const EditUserProfileImg = () => {

  const { data: session, update } = useSession();
  const [image, setImage] = useState<string>('');
  const [message, setMessage] = useState<string>('');
  const [plotlyHTML, setPlotlyHTML] = useState<string>('');
  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null);
  useEffect(() => { console.log("session update") }, [session]);


  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImagePreview({ url, file });
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!imagePreview) {
      setMessage('Please select an image.');
      return;
    }

    try {
      if (imagePreview) {
        const formData = new FormData();
        formData.append('file', imagePreview.file);
        console.log(imagePreview.file);
        const response = await fetch(
          `${process.env.Localhost}/user/modification/profile_image`,
          {
            method: 'PUT',
            headers: {
              Authorization: `Bearer ${(session as any)?.access_token}`,
            },
            body: formData,
          }
        );
        session.user = await response.json();
        update(session);

      }
    } catch (error) {
      console.error('Error updating profile image:', error);
      setMessage('An error occurred while updating profile image.');
    }
  };

  return (
    <div className='absolute left-[250px] top-[600px]'>
      {message && <p>{message} </p>}
      <form onSubmit={handleSubmit} >
        <input type="file" accept="image/*" onChange={handleImageChange} />
        {image && <img src={imagePreview.url} alt="Preview" />}
        <button className='absolute left-[40px] top-[90px] justify-center rounded-md bg-yellow-700 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500'
          type="submit" > 프로필 사진 저장 </button>
      </form>
    </div>
    // <div className='absolute left-[250px] top-[600px]'>
    //   {message && <p>{message}</p>}
    //   <form onSubmit={handleSubmit}>
    //     <input type="file" accept="image/*" onChange={handleImageChange} />
    //     {imagePreview && <img src={imagePreview.url} alt="Preview" />}
    //     <button className='absolute left-[40px] top-[90px] justify-center rounded-md bg-yellow-700 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500' type="submit">프로필 사진 저장</button>
    //   </form>
    // </div>
  );
};

export default EditUserProfileImg;
