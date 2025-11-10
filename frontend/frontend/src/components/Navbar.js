import React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
// import IconButton from '@mui/material/IconButton';
// import MenuIcon from '@mui/icons-material/Menu';
import { Link } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" color="primary" sx={{ borderRadius: 0, boxShadow: 2 }}>
        <Toolbar>
          {/* <IconButton size="large" edge="start" color="inherit" aria-label="menu" sx={{ mr: 2 }}>
            <MenuIcon />
          </IconButton> */}
          <img src="/logo192.png" alt="logo" style={{ width: 40, marginRight: 12 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 700 }}>
            COE Project
          </Typography>
          <Link className="nav-link" to="/">Home</Link>
          <Link className="nav-link" to="/upload">Upload</Link>
          <Link className="nav-link" to="/generate">Generate</Link>
          <Link className="nav-link" to="/search">Search</Link>
        </Toolbar>
      </AppBar>
    </Box>
  );
}