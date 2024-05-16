import React, { useState } from "react";
import Image from "next/image";
import styles from "./Modal.module.css";

const Modal = ({
  imageUrl,
  imageLanding,
  onClose,
}: {
  imageUrl: string;
  imageLanding: string;
  onClose: () => void;
}) => {
  const [modalOpen, setModalOpen] = useState(true);

  const LinkBtClick = () => {
    window.open(imageLanding, "_blank");
  };

  const closeModal = () => {
    setModalOpen(false);
    setTimeout(() => {
      onClose();
    }, 300);
  };

  return (
    <div
      className={`${styles.modalOverlay} ${modalOpen ? styles.active : ""}`}
      onClick={closeModal}
    >
      <div className={`${styles.modal} ${modalOpen ? styles.active : ""}`}>
        <Image
          src={imageUrl}
          alt="Preview Image"
          layout="responsive"
          width={800}
          height={600}
        />
      </div>
      <button onClick={LinkBtClick} className="mt-3 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full">링크 가기</button>
    </div>
  );
};

export default Modal;
