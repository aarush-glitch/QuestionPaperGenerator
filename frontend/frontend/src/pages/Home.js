import React from 'react';

export default function Home() {
  return (
    <div className="section">
      <h1>Welcome to the COE Project</h1>
      <p>
        This platform allows you to extract text from PDFs/images, generate questions, clean subtopics, embed questions for semantic search, and search/filter questions with ease.
      </p>
      <ul>
        <li>Upload your PDF or image files</li>
        <li>Generate questions from extracted content</li>
        <li>Clean and standardize subtopics</li>
        <li>Embed and search questions semantically</li>
      </ul>
    </div>
  );
}