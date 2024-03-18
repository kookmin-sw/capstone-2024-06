"use client";
import React, { useState } from "react";

interface Props {
  OnSelectCategory: (category: string) => void;
};

const SelectCategory: React.FC<Props> = ({ OnSelectCategory }) => {
  const [SelectCategorys, SetSelectCategorys] = useState("자유");

  const CategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    SetSelectCategorys(e.target.value);
    OnSelectCategory(e.target.value);
  };

  return (
    <main>
      <form className="max-w-sm mx-auto">
        <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">
          카테고리
        </label>
        <select
          id="countries"
          value={SelectCategorys}
          onChange={CategoryChange}
          className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
        >
          <option value="자유">자유</option>
          <option value="팝니다">팝니다</option>
          <option value="삽니다">삽니다</option>
          <option value="인기">인기</option>
          <option value="실시간">실시간</option>
        </select>
      </form>
    </main>
  );
};

export default SelectCategory;
