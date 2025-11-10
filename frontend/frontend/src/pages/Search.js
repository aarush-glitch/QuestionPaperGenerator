import React, { useState } from 'react';
import axios from 'axios';

export default function Search() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    setStatus('Searching...');
    try {
      const res = await axios.post('/api/search', { query });
      setResults(res.data.results || []);
      setStatus('');
    } catch (err) {
      setStatus('Search failed.');
    }
  };

  return (
    <div className="section">
      <h2>Semantic Search</h2>
      <form onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Enter your search query..."
          style={{ width: 300 }}
        />
        <button type="submit" style={{ marginLeft: 12 }}>Search</button>
      </form>
      <div style={{ marginTop: 16 }}>{status}</div>
      <ul style={{ marginTop: 24 }}>
        {results.map((item, idx) => (
          <li key={idx} style={{ marginBottom: 12 }}>
            <strong>Q:</strong> {item.question}<br />
            <span style={{ color: '#888' }}>Topic: {item.topic}, Subtopic: {item.subtopic}, Marks: {item.marks}, Difficulty: {item.difficulty}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}