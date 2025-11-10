import React, { useState } from 'react';
import axios from 'axios';

export default function Upload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setStatus('Uploading...');
    try {
      const res = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStatus(res.data.message || 'Upload successful!');
    } catch (err) {
      setStatus('Upload failed.');
    }
  };

  return (
    <div className="section">
      <h2>Upload PDF or Image</h2>
      <form onSubmit={handleUpload}>
        <input type="file" accept=".pdf,image/*" onChange={handleFileChange} />
        <button type="submit" style={{ marginLeft: 12 }}>Upload</button>
      </form>
      <div style={{ marginTop: 16 }}>{status}</div>
    </div>
  );
}