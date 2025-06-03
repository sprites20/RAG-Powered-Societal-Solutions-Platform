import React, { useState } from 'react';
import './BoxList.css';

// Dummy job data for demonstration
const jobData = Array.from({ length: 50 }, (_, i) => ({
  id: i + 1,
  title: `Job Title ${i + 1}: Software Engineer`,
  description: `This is a detailed description for Job ${i + 1}. We are looking for a passionate software engineer to join our dynamic team. Responsibilities include developing and maintaining web applications, collaborating with cross-functional teams, and contributing to all phases of the development lifecycle. Experience with React, Node.js, and cloud platforms is a plus. This role offers excellent growth opportunities and a collaborative work environment. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.`,
  location: `City ${Math.floor(Math.random() * 5) + 1}, State`,
  // Add more details as needed
}));

const truncateDescription = (text, maxLength) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

const BoxList = () => {
  const jobsPerPage = 9; // jobs to show per page
  const [currentPage, setCurrentPage] = useState(1);
  const [uploadedFiles, setUploadedFiles] = useState([]); // State to store uploaded files for the whole list

  const totalPages = Math.ceil(jobData.length / jobsPerPage);

  const startIdx = (currentPage - 1) * jobsPerPage;
  const visibleJobs = jobData.slice(startIdx, startIdx + jobsPerPage);

  const handleNext = () => {
    if (currentPage < totalPages) setCurrentPage(currentPage + 1);
  };

  const handlePrev = () => {
    if (currentPage > 1) setCurrentPage(currentPage - 1);
  };

  const handleJobClick = (jobId) => {
    alert(`Clicked on Job ID: ${jobId}`);
    // In a real application, you would navigate to a job detail page
  };

  const handleDragOver = (event) => {
    event.preventDefault(); // Prevent default to allow drop
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) {
      setUploadedFiles(files.map(file => file.name));
      alert(`File(s) dropped: ${files.map(file => file.name).join(', ')}`);
    }
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      setUploadedFiles(files.map(file => file.name));
      alert(`File(s) selected: ${files.map(file => file.name).join(', ')}`);
    }
  };

  const triggerFileInput = () => {
    document.getElementById('fileInput').click();
  };

  return (
    <div className="box-list-container">
      <div 
        className="overall-drop-area"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        Drag & Drop Your Resume/Files Here
        <p>or</p>
        <button onClick={triggerFileInput} className="select-file-button">
          Select Files
        </button>
        <input 
          type="file" 
          id="fileInput" 
          style={{ display: 'none' }} 
          onChange={handleFileSelect} 
          multiple 
        />
        {uploadedFiles.length > 0 && (
          <p className="overall-uploaded-file-info">Uploaded: {uploadedFiles.join(', ')}</p>
        )}
      </div>

      <div className="boxes-frame">
        {visibleJobs.map((job) => (
          <div key={job.id} className="box" onClick={() => handleJobClick(job.id)}>
            <h3 className="job-title">{job.title}</h3>
            <p className="job-description">{truncateDescription(job.description, 150)}</p>
            <p className="job-location">Location: {job.location}</p>
            {/* Add more job details here */}
          </div>
        ))}
      </div>

      <div className="pagination">
        <button
          onClick={handlePrev}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        <span>Page {currentPage} of {totalPages}</span>
        <button
          onClick={handleNext}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default BoxList;
