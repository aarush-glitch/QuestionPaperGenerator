import React, { useState } from 'react';
import axios from 'axios';

export default function Generate() {
  const [status, setStatus] = useState('');

  const handleGenerate = async () => {
    setStatus('Generating questions...');
    try {
      const res = await axios.post('/api/generate-questions');
      setStatus(res.data.message || 'Questions generated!');
    } catch (err) {
      setStatus('Generation failed.');
    }
  };

  const handleClean = async () => {
    setStatus('Cleaning subtopics...');
    try {
      const res = await axios.post('/api/clean-subtopics');
      setStatus(res.data.message || 'Subtopics cleaned!');
    } catch (err) {
      setStatus('Cleaning failed.');
    }
  };

  const handleEmbed = async () => {
    setStatus('Embedding questions...');
    try {
      const res = await axios.post('/api/embed');
      setStatus(res.data.message || 'Embedding complete!');
    } catch (err) {
      setStatus('Embedding failed.');
    }
  };

  return (
    <div className="section">
      <h2>Process Extracted Content</h2>
      <button onClick={handleGenerate}>Generate Questions</button>
      <button onClick={handleClean} style={{ marginLeft: 12 }}>Clean Subtopics</button>
      <button onClick={handleEmbed} style={{ marginLeft: 12 }}>Embed Questions</button>
      <div style={{ marginTop: 16 }}>{status}</div>
    </div>
  );
}