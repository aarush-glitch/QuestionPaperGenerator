



import React, { useState, useRef } from 'react';
import axios from 'axios';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CleaningServicesIcon from '@mui/icons-material/CleaningServices';
import StorageIcon from '@mui/icons-material/Storage';
import SearchIcon from '@mui/icons-material/Search';
import DownloadIcon from '@mui/icons-material/Download';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { styled } from '@mui/material/styles';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Container from '@mui/material/Container';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import CssBaseline from '@mui/material/CssBaseline';
import { motion, AnimatePresence } from 'framer-motion';


const drawerWidth = 270;
const steps = [
  { label: '1) OCR & Refinement', icon: <UploadFileIcon /> },
  { label: '2) Subtopic Cleanup', icon: <CleaningServicesIcon /> },
  { label: '3) Build Vector Store (FAISS)', icon: <StorageIcon /> },
  { label: '4) Search & Smart Filter', icon: <SearchIcon /> },
  { label: '5) Export Results', icon: <DownloadIcon /> },
];

const GradientAppBar = styled(AppBar)(({ theme }) => ({
  background: 'linear-gradient(90deg, #3338A0 0%, #C59560 100%)',
  boxShadow: '0 4px 24px 0 rgba(51,56,160,0.10)',
  minHeight: 64,
  justifyContent: 'center',
}));

const SidebarListItemButton = styled(ListItemButton)(({ theme }) => ({
  borderRadius: 6,
  margin: '3px 6px',
  transition: 'background 0.2s, box-shadow 0.2s',
  fontSize: '0.89rem',
  paddingTop: 4,
  paddingBottom: 4,
  '&.Mui-selected': {
    background: '#FCC61D', // highlight selected with palette yellow
    color: '#3338A0',
    fontWeight: 700,
    '& .MuiListItemIcon-root': {
      color: '#3338A0',
    },
  },
  '&:hover': {
    background: '#C59560', // hover with palette accent
    color: '#fff',
    '& .MuiListItemIcon-root': {
      color: '#fff',
    },
  },
}));

const AnimatedPaper = styled(motion.div)({
  width: '100%',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  alignItems: 'center',
});


function App() {
  const [selectedStep, setSelectedStep] = useState(0);
  // State for each step
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [ocrPreview, setOcrPreview] = useState('');
  const [generateStatus, setGenerateStatus] = useState('');
  const [cleanStatus, setCleanStatus] = useState('');
  const [embedStatus, setEmbedStatus] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMarks, setSearchMarks] = useState('');
  const [searchDifficulty, setSearchDifficulty] = useState('');
  const [searchCognitive, setSearchCognitive] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchStatus, setSearchStatus] = useState('');
  const fileInputRef = useRef();

  // Handlers for each step
  const handleFileChange = (e) => {
    setUploadFile(e.target.files[0]);
  };
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;
    setUploadStatus('Uploading...');
    setOcrPreview('');
    const formData = new FormData();
    formData.append('file', uploadFile);
    try {
      const res = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploadStatus(res.data.message || 'Upload successful!');
      setOcrPreview(res.data.extracted_text || '');
    } catch (err) {
      setUploadStatus('Upload failed.');
      setOcrPreview('');
    }
  };
  const handleGenerate = async () => {
    setGenerateStatus('Generating questions...');
    try {
      const res = await axios.post('/api/generate-questions');
      setGenerateStatus(res.data.message || 'Questions generated!');
    } catch (err) {
      setGenerateStatus('Generation failed.');
    }
  };
  const handleClean = async () => {
    setCleanStatus('Cleaning subtopics...');
    try {
      const res = await axios.post('/api/clean-subtopics');
      setCleanStatus(res.data.message || 'Subtopics cleaned!');
    } catch (err) {
      setCleanStatus('Cleaning failed.');
    }
  };
  const handleEmbed = async () => {
    setEmbedStatus('Embedding questions...');
    try {
      const res = await axios.post('/api/embed');
      setEmbedStatus(res.data.message || 'Embedding complete!');
    } catch (err) {
      setEmbedStatus('Embedding failed.');
    }
  };
  const handleSearch = async (e) => {
    e.preventDefault();
    setSearchStatus('Searching...');
    setSearchResults([]);
    try {
      const res = await axios.post('/api/search', {
        query: searchQuery,
        marks: searchMarks ? parseInt(searchMarks) : undefined,
        difficulty: searchDifficulty,
        cognitive: searchCognitive,
      });
      setSearchResults(res.data.results || []);
      setSearchStatus('');
    } catch (err) {
      setSearchStatus('Search failed.');
      setSearchResults([]);
    }
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', background: 'linear-gradient(135deg, #e0e7ff 0%, #fff 100%)' }}>
      <CssBaseline />
      <GradientAppBar position="fixed" sx={{ zIndex: 1300 }}>
        <Toolbar sx={{ minHeight: 64, display: 'flex', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <AutoAwesomeIcon sx={{ mr: 2, fontSize: 28, color: '#FCC61D' }} />
            <Typography variant="h6" noWrap sx={{ fontWeight: 700, letterSpacing: 0.5, color: '#F7F7F7', fontSize: '1.15rem' }}>
              LLM Question Paper Pipeline
            </Typography>
          </Box>
        </Toolbar>
      </GradientAppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: 'border-box',
            background: '#F7F7F7',
            borderRight: 0,
            boxShadow: '2px 0 16px 0 rgba(51,56,160,0.07)',
            borderTopRightRadius: 0,
            borderBottomRightRadius: 0,
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', pt: 2 }}>
          <List>
            {steps.map((step, idx) => (
              <ListItem key={step.label} disablePadding>
                <SidebarListItemButton selected={selectedStep === idx} onClick={() => setSelectedStep(idx)}>
                  <ListItemIcon sx={{ minWidth: 40 }}>{step.icon}</ListItemIcon>
                  <ListItemText primary={step.label} primaryTypographyProps={{ fontWeight: selectedStep === idx ? 700 : 500, fontSize: 17 }} />
                </SidebarListItemButton>
              </ListItem>
            ))}
          </List>
          <Divider sx={{ my: 2 }} />
          <Box px={2}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>Environment Notes</Typography>
            <Typography variant="body2" color="text.secondary">
              Requires: pytesseract, tesseract, PyMuPDF, opencv-python, langchain, faiss-cpu, Ollama.<br />
              Ollama must be running locally.<br />
              Ensure questions.json format is correct.
            </Typography>
          </Box>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 0, ml: `${drawerWidth}px`, display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'linear-gradient(135deg, #F7F7F7 0%, #fff 100%)' }}>
        <Toolbar />
        <Container maxWidth={false} sx={{ flex: 1, py: 4, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <AnimatePresence mode="wait">
            {selectedStep === 0 && (
              <AnimatedPaper
                key="ocr"
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -40, scale: 0.98 }}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                style={{ minHeight: 400 }}
              >
                <Paper elevation={4} sx={{ p: 3, borderRadius: 3, width: '100%', maxWidth: 900, background: 'rgba(255,255,255,0.97)' }}>
                  <Typography variant="h6" fontWeight={700} mb={1.5} color="#3338A0">1) OCR & Refinement</Typography>
                  <Typography mb={1.5} fontSize="0.97rem">Upload an <b>image</b> or <b>PDF</b>. We'll run OCR, basic refinement, and a light analysis as implemented in your backend.</Typography>
                  <form onSubmit={handleUpload} style={{ marginBottom: 16 }}>
                    <input type="file" accept=".pdf,image/*" onChange={handleFileChange} ref={fileInputRef} />
                    <button type="submit" style={{ marginLeft: 12 }}>Upload & OCR</button>
                  </form>
                  <div style={{ marginBottom: 8, color: '#3338A0' }}>{uploadStatus}</div>
                  {ocrPreview && (
                    <Paper elevation={1} sx={{ p: 2, mt: 1, background: '#f5f7fa', fontSize: '0.95rem', maxHeight: 200, overflow: 'auto' }}>
                      <b>Preview:</b>
                      <div style={{ whiteSpace: 'pre-wrap' }}>{ocrPreview}</div>
                    </Paper>
                  )}
                </Paper>
              </AnimatedPaper>
            )}
            {selectedStep === 1 && (
              <AnimatedPaper
                key="subtopic"
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -40, scale: 0.98 }}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                style={{ minHeight: 400 }}
              >
                <Paper elevation={4} sx={{ p: 3, borderRadius: 3, width: '100%', maxWidth: 900, background: 'rgba(255,255,255,0.97)' }}>
                  <Typography variant="h6" fontWeight={700} mb={1.5} color="#3338A0">2) Subtopic Cleanup</Typography>
                  <Typography mb={1.5} fontSize="0.97rem">Clean and normalize <b>subtopic</b> fields in your question JSON using backend rules.</Typography>
                  <button onClick={handleClean}>Clean Subtopics</button>
                  <div style={{ marginTop: 16, color: '#3338A0' }}>{cleanStatus}</div>
                </Paper>
              </AnimatedPaper>
            )}
            {selectedStep === 2 && (
              <AnimatedPaper
                key="vector"
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -40, scale: 0.98 }}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                style={{ minHeight: 400 }}
              >
                <Paper elevation={4} sx={{ p: 3, borderRadius: 3, width: '100%', maxWidth: 900, background: 'rgba(255,255,255,0.97)' }}>
                  <Typography variant="h6" fontWeight={700} mb={1.5} color="#3338A0">3) Build Vector Store (FAISS)</Typography>
                  <Typography mb={1.5} fontSize="0.97rem">Embed questions using Ollama and store them in a FAISS index.</Typography>
                  <button onClick={handleEmbed}>Embed Questions</button>
                  <div style={{ marginTop: 16, color: '#3338A0' }}>{embedStatus}</div>
                </Paper>
              </AnimatedPaper>
            )}
            {selectedStep === 3 && (
              <AnimatedPaper
                key="search"
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -40, scale: 0.98 }}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                style={{ minHeight: 400 }}
              >
                <Paper elevation={4} sx={{ p: 3, borderRadius: 3, width: '100%', maxWidth: 900, background: 'rgba(255,255,255,0.97)' }}>
                  <Typography variant="h6" fontWeight={700} mb={1.5} color="#3338A0">4) Search & Smart Filter</Typography>
                  <Typography mb={1.5} fontSize="0.97rem">Query the FAISS index, then filter by marks, difficulty, and cognitive level.</Typography>
                  <form onSubmit={handleSearch} style={{ marginBottom: 16 }}>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder="Enter your search query..."
                      style={{ width: 300 }}
                    />
                    <input
                      type="number"
                      value={searchMarks}
                      onChange={e => setSearchMarks(e.target.value)}
                      placeholder="Marks"
                      style={{ width: 80, marginLeft: 8 }}
                    />
                    <input
                      type="text"
                      value={searchDifficulty}
                      onChange={e => setSearchDifficulty(e.target.value)}
                      placeholder="Difficulty"
                      style={{ width: 120, marginLeft: 8 }}
                    />
                    <input
                      type="text"
                      value={searchCognitive}
                      onChange={e => setSearchCognitive(e.target.value)}
                      placeholder="Cognitive Level"
                      style={{ width: 140, marginLeft: 8 }}
                    />
                    <button type="submit" style={{ marginLeft: 12 }}>Search</button>
                  </form>
                  <div style={{ marginBottom: 8, color: '#3338A0' }}>{searchStatus}</div>
                  <ul style={{ marginTop: 24 }}>
                    {searchResults.map((item, idx) => (
                      <li key={idx} style={{ marginBottom: 12 }}>
                        <strong>Q:</strong> {item.question}<br />
                        <span style={{ color: '#888' }}>Topic: {item.topic}, Subtopic: {item.subtopic}, Marks: {item.marks}, Difficulty: {item.difficulty}, Cognitive: {item.cognitive_level}</span>
                      </li>
                    ))}
                  </ul>
                </Paper>
              </AnimatedPaper>
            )}
            {selectedStep === 4 && (
              <AnimatedPaper
                key="export"
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -40, scale: 0.98 }}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                style={{ minHeight: 400 }}
              >
                <Paper elevation={4} sx={{ p: 3, borderRadius: 3, width: '100%', maxWidth: 900, background: 'rgba(255,255,255,0.97)' }}>
                  <Typography variant="h6" fontWeight={700} mb={1.5} color="#3338A0">5) Export Results</Typography>
                  <Typography mb={1.5} fontSize="0.97rem">Download refined text and/or the search results.</Typography>
                  {/* TODO: Download buttons for refined text and search results */}
                </Paper>
              </AnimatedPaper>
            )}
          </AnimatePresence>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
