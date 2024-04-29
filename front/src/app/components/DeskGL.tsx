"use client";
import React, { useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera, Environment, OrbitControls, useGLTF, Plane } from "@react-three/drei";
import * as THREE from 'three';


const DeskGL = () => {
  const DeskModel = useGLTF("./models/desk.glb");
  const ChairModel = useGLTF("./models/chair.glb");
  const BooksModel = useGLTF("./models/books.glb");

  const BookRef = useRef<THREE.Object3D | null>(null);
  const [BookModelV, SetBookModelV] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [offset, setOffset] = useState([0, 0]);

  const handlePointerDown = (event : any) => {
    setIsDragging(true);
    const x = (event.clientX / window.innerWidth) * 2 - 1;
    const z = -(event.clientY / window.innerHeight) * 2 + 1;
    setOffset([x, z]);
  };

  const handlePointerMove = (event : any) => {
    if (isDragging && BookRef.current) {
      const x = (event.clientX / window.innerWidth) * 2 - 1;
      const Z = -(event.clientY / window.innerHeight) * 2 + 1;
      BookRef.current.position.x = x - offset[0];
      BookRef.current.position.z = Z - offset[1];
    }
  };

  const handlePointerUp = () => {
    setIsDragging(false);
  };

  return (
    <main>
      <div className="flex-col">
        <button onClick={() => SetBookModelV(!BookModelV)} className="border">
          Book
        </button>
        <Canvas style={{ width: "", height: 600 }} className="border" onPointerMove={handlePointerMove} onPointerUp={handlePointerUp}>
          <Environment preset="apartment" blur={0.5} />
          <PerspectiveCamera makeDefault position={[5, 5, 5]} />
          {!isDragging && <OrbitControls />}
          <Plane args={[10, 10]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]} />
          <primitive object={DeskModel.scene} position={[0, -1, 0]} />
          <primitive object={ChairModel.scene} position={[1, -1, -0.3]} rotation={[0, -(Math.PI / 2), 0]} />
          {BookModelV && <primitive object={BooksModel.scene} position={[0, 0, 0]} ref={BookRef} scale={[0.5,0.5,0.5]} onPointerDown={handlePointerDown} />}
        </Canvas>
      </div>
    </main>
  );
};

export default DeskGL;
