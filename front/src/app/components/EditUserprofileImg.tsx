import { useState } from 'react';
import { useSession } from 'next-auth/react';

const EditUserProfileImg = () => {
  const { data: session } = useSession();
  const [image, setImage] = useState(null);
  const [message, setMessage] = useState('');

  const handleImageChange = (e) => {
    const selectedImage = e.target.files[0];
    setImage(selectedImage);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setMessage('Please select an image.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('/api/user/image', {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${session.accessToken}`,
        },
        body: formData,
      });
      if (response.ok) {
        setMessage('Profile image updated successfully.');
      } else {
        const data = await response.json();
        setMessage(data.error || 'Failed to update profile image.');
      }
    } catch (error) {
      console.error('Error updating profile image:', error);
      setMessage('An error occurred while updating profile image.');
    }
  };

  return (
    <div>
    { message && <p>{ message } < /p>}
    < form onSubmit = { handleSubmit } >
      <input type="file" accept = "image/*" onChange = { handleImageChange } />
        { image && <img src={ URL.createObjectURL(image) } alt = "Preview" />}
<button type="submit" > Save Changes < /button>
  < /form>
  < /div>
  );
};

export default EditUserProfileImg;
