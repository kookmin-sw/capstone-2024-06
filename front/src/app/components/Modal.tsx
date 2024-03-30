import React, { useState } from "react";
import Image from "next/image";
import styles from "./Modal.module.css";

const Modal = ({
  imageUrl,
  onClose,
}: {
  imageUrl: string;
  onClose: () => void;
}) => {
  const [modalOpen, setModalOpen] = useState(true);

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
    </div>
  );
};

export default Modal;
