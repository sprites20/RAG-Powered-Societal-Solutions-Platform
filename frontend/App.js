import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { Container, CssBaseline, Box } from '@mui/material';

// Importing components
import Navbar from './Navbar';
import Home from './Home';
import CategoryPage from './CategoryPage';

// Main App Component
function App() {
  return (
    <Router>
      <CssBaseline />
      {/* Navbar at the top */}
      <Navbar />

      {/* Main content */}
      <Box sx={{ paddingTop: 10 }}>
        <Routes>
          {/* Home Page */}
          <Route path="/" element={<Home />} />

          {/* Category Pages */}
		  
		  <Route path="/category/general" element={<CategoryPage category="General" />} />
		  
          <Route path="/category/basic-infrastructure" element={<CategoryPage category="Basic Infrastructure" />} />
		  <Route path="/category/basic-infrastructure/electricity" element={<CategoryPage category="Electricity" />} />
		  <Route path="/category/basic-infrastructure/water-supply" element={<CategoryPage category="Water Supply" />} />
		  
          <Route path="/category/economic-systems" element={<CategoryPage category="Economic and Industrial Systems" />} />
          <Route path="/category/health-and-safety" element={<CategoryPage category="Health and Safety" />} />
          <Route path="/category/social-and-cultural-systems" element={<CategoryPage category="Social and Cultural Systems" />} />
          <Route path="/category/environmental-management" element={<CategoryPage category="Environmental Management" />} />
          <Route path="/category/governance-and-administration" element={<CategoryPage category="Governance and Administration" />} />
		  <Route path="/category/other-critical-aspects" element={<CategoryPage category="Other Critical Aspects" />} />
		  
		  {/* Add more categories as needed */}
        </Routes>
      </Box>
    </Router>
  );
}

export default App;
