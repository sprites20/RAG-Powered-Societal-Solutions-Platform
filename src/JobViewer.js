import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import axios from 'axios';
import './JobViewer.css';

function JobViewer() {
  const [searchParams] = useSearchParams();
  const id = searchParams.get('id');
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (!id) return;
    axios
      .get(`http://localhost:5000/job?id=${id}`)
      .then((res) => setJob(res.data))
      .catch((err) =>
        setError(err.response?.data?.error || 'Failed to load job data')
      );
  }, [id]);

  if (!id) return <div>Please provide a job ID in the URL (e.g. /job?id=123)</div>;
  if (error) return <div>Error: {error}</div>;
  if (!job) return <div>Loading...</div>;

  return (
    <div className="job-viewer-container">
      <h1 className="job-viewer-title">{job.title}</h1>
      <p className="job-viewer-text">
        <span className="job-viewer-label">Company:</span> {job.company_name}
      </p>
      <p className="job-viewer-text">
        <span className="job-viewer-label">Location:</span> {job.location}
      </p>
      <p className="job-viewer-text">
        <span className="job-viewer-label">Description:</span> {job.description}
      </p>
      <p className="job-viewer-text">
        <span className="job-viewer-label">Skills:</span> {job.skills_desc}
      </p>
      <p className="job-viewer-text">
        <span className="job-viewer-label">Work Type:</span> {job.work_type}
      </p>
      <p className="job-viewer-text">
        <span className="job-viewer-label">Salary:</span> {job.min_salary}–{job.max_salary} {job.currency}
      </p>
    </div>
  );
}

export default JobViewer;
