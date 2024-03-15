"use client";
import React, { useState } from "react";

const SelectCategory = () => {
  return (
    <main>
      <form className="max-w-sm mx-auto">
        <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">
          카테고리
        </label>
        <select
          id="countries"
          className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
        >
          <option selected>자유</option>
          <option value="">팝니다</option>
          <option value="">삽니다</option>
          <option value="">인기</option>
          <option value="">실시간</option>
        </select>
      </form>
    </main>
  );
};


export default SelectCategory;