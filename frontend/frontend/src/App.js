import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Upload from './pages/Upload';
import Generate from './pages/Generate';
import Search from './pages/Search';
import './App.css';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/generate" element={<Generate />} />
        <Route path="/search" element={<Search />} />
      </Routes>
    </Router>
  );
}

export default App;