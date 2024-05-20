"use client";
import React, { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  PerspectiveCamera,
  Environment,
  OrbitControls,
  useGLTF,
  Plane,
} from "@react-three/drei";
import * as THREE from "three";

const Arrow = ({ position }: { position: number[] }) => {
  const ArrowModel = useGLTF("./models/arrow.glb");
  const arrowRef = useRef();
  var [yPos, setYPos] = useState(0);

  useFrame(() => {
    if (arrowRef.current) {
      //@ts-ignore
      arrowRef.current.rotation.y += 0.01;
      //@ts-ignore
      if (arrowRef.current.rotation.y >= Math.PI * 2) {
        //@ts-ignore
        arrowRef.current.rotation.y = 0;
      }
      yPos += 0.1;
      yPos -= 0.1;
    }
  });

  return (
    <primitive
      object={ArrowModel.scene}
      position={[position[0], yPos, position[2]]}
      scale={[0.05, 0.05, 0.05]}
      ref={arrowRef}
      rotation={[0, -(Math.PI / 2), -(Math.PI / 2)]}
    />
  );
};

const DeskGL = () => {
  const DeskModel = useGLTF("./models/desk.glb");
  const ChairModel = useGLTF("./models/chair.glb");
  const BooksModel = useGLTF("./models/books.glb");
  const PlantModel = useGLTF("./models/plant.glb");

  const [ModelClicked, SetModelClicked] = useState(false);

  const [models, setModels] = useState([
    {
      id: "book",
      model: BooksModel,
      visible: false,
      ref: useRef<THREE.Object3D | null>(null),
      scale: [0.5, 0.5, 0.5],
      position: [0, -0.11, 0],
      click: false,
    },
    {
      id: "plant",
      model: PlantModel,
      visible: false,
      ref: useRef<THREE.Object3D | null>(null),
      scale: [0.1, 0.1, 0.1],
      position: [0, -0.19, 0],
      click: false,
    },
  ]);

  const ModelClick = () => {
    SetModelClicked(!ModelClicked);
  };

  const toggleModelVisibility = (id: string) => {
    setModels((prevModels) =>
      prevModels.map((model) =>
        model.id === id ? { ...model, visible: !model.visible } : model
      )
    );
  };

  const ArrowPosition = [[-0.4, 0, 0], [-0.4, 0, -0.2]];

  return (
    <main>
      <div className="flex">
        <Canvas style={{ width: "100%", height: 600 }} className="border">
          <Environment preset="apartment" blur={0.5} />
          <PerspectiveCamera makeDefault position={[5, 5, 5]} />
          <OrbitControls />
          <Plane
            args={[10, 10]}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, -1, 0]}
          />
          <primitive object={DeskModel.scene} position={[0, -1, 0]} />
          <primitive
            object={ChairModel.scene}
            position={[1, -1, -0.3]}
            rotation={[0, -(Math.PI / 2), 0]}
          />
          {!ModelClicked && ArrowPosition.map((position) => <Arrow position={position} />)}
          {models.map(
            (model) =>
              model.visible && (
                <primitive
                  key={model.id}
                  object={model.model.scene}
                  ref={model.ref}
                  scale={model.scale}
                  position={model.position}
                  onClick={() => ModelClick()}
                />
              )
          )}
        </Canvas>
        <div className="flex-wrap border w-[300px] h-fit p-4">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => toggleModelVisibility(model.id)}
              className="border w-1/3"
            >
              {model.id}
            </button>
          ))}
        </div>
      </div>
    </main>
  );
};

export default DeskGL;
