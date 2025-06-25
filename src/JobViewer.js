import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './JobViewer.css';


function JobViewer() {
  const [searchParams] = useSearchParams();
  const id = searchParams.get('id');
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [deviceId, setDeviceId] = useState("");
  const [resumeTokens, setResumeTokens] = useState([]);
  const [resumeText, setResumeText] = useState("");

  const [aiResponses, setAiResponses] = useState({
    job_fit: null,
    tips: null,
    skill_gap: null,
    legitimacy: null,
  });
  
  const [loading, setLoading] = useState({
    job_fit: false,
    tips: false,
    skill_gap: false,
    legitimacy: false,
  });
  const handleExpand = (type) => {
    if (!aiResponses[type] && !loading[type]) {
      fetchAIResponse(type, job, resumeText);
    }
  };
  const fetchAIResponse = async (type, job, resumeText) => {
    setLoading(prev => ({ ...prev, [type]: true }));
  
    let prompt = '';
    switch (type) {
      case 'job_fit':
        prompt = `Based on this job description and my resume tokens, do I fit this role?\n\nJob:\n${job.description}\n\nResume Tokens:\n${resumeText}`;
        break;
      case 'tips':
        prompt = `Give application tips for this job:\n\n${job.description} with this other info ${job.skills_desc}`;
        break;
      case 'skill_gap':
        prompt = `Identify skill gaps between my resume and this job, and recommend online courses.\n\nJob:\n${job.description}\n\nResume Tokens:\n${resumeText}`;
        break;
      case 'legitimacy':
        prompt = `Analyze whether this job posting seems legitimate:\n\nCompany: ${job.company_name}\n\nDescription:\n${job.description}\n\nSkills:\n${job.skills_desc}}`;
        break;
    }
  
    try {
      const res = await fetch("http://localhost:5000/api/ai_recommendation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
  
      const result = await res.json();
      setAiResponses(prev => ({ ...prev, [type]: result.result || "No response." }));
    } catch (err) {
      setAiResponses(prev => ({ ...prev, [type]: "Failed to fetch AI recommendation." }));
    } finally {
      setLoading(prev => ({ ...prev, [type]: false }));
    }
  };
  
  
  function getCookie(name) {
    return document.cookie
      .split("; ")
      .reduce((r, v) => {
        const parts = v.split("=");
        return parts[0] === name ? decodeURIComponent(parts[1]) : r;
      }, "");
  }

  function buildBadCharTable(pattern) {
    const table = new Array(256).fill(-1);
    for (let i = 0; i < pattern.length; i++) {
      table[pattern.charCodeAt(i)] = i;
    }
    return table;
  }
  
  function boyerMooreSearch(text, pattern) {
    const matches = [];
    const badChar = buildBadCharTable(pattern);
    let shift = 0;
  
    const m = pattern.length;
    const n = text.length;
  
    while (shift <= n - m) {
      let j = m - 1;
  
      while (j >= 0 && pattern[j].toLowerCase() === text[shift + j].toLowerCase()) {
        j--;
      }
  
      if (j < 0) {
        matches.push(shift);
        shift += (shift + m < n) ? m - badChar[text.charCodeAt(shift + m)] || 1 : 1;
      } else {
        shift += Math.max(1, j - badChar[text.charCodeAt(shift + j)] || -1);
      }
    }
  
    return matches;
  }
  function highlightDescription(description, tokens) {
    if (!tokens || tokens.length === 0) return description;
  
    // Escape special characters for regex
    const escapedTokens = tokens.map(token => token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  
    const regex = new RegExp(`\\b(${escapedTokens.join('|')})\\b`, 'gi');
    return description.replace(regex, '<mark>$1</mark>');
  }

  function highlightDescriptionWithBoyerMoore(description, tokens) {
    if (!tokens || tokens.length === 0) return description;
  
    const lowerDesc = description.toLowerCase();
    const markPositions = [];
  
    for (let token of tokens) {
      token = token.trim();
      if (!token) continue;
  
      const positions = boyerMooreSearch(lowerDesc, token.toLowerCase());
      for (let pos of positions) {
        markPositions.push({ start: pos, end: pos + token.length });
      }
    }
  
    // Sort and merge overlapping highlights
    markPositions.sort((a, b) => a.start - b.start);
    const merged = [];
    for (let pos of markPositions) {
      if (!merged.length || merged[merged.length - 1].end < pos.start) {
        merged.push(pos);
      } else {
        merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, pos.end);
      }
    }
  
    // Build final string with <mark>
    let result = '';
    let lastIndex = 0;
    for (let { start, end } of merged) {
      result += description.slice(lastIndex, start);
      result += `<mark>${description.slice(start, end)}</mark>`;
      lastIndex = end;
    }
    result += description.slice(lastIndex);
  
    return result;
  }
  useEffect(() => {
    if (!id) return;
    axios
      .get(`http://localhost:5000/job?id=${id}`)
      .then((res) => setJob(res.data))
      .catch((err) =>
        setError(err.response?.data?.error || 'Failed to load job data')
      );
  }, [id]);
  // Set device UUID cookie on first load
  useEffect(() => {
    let uuid = getCookie("device_uuid");
    if (!uuid) {
      console.log("No device UUID found");
    }
    else{
      setDeviceId(uuid);
      console.log("Device UUID:", deviceId);
    }
  }, []);

  useEffect(() => {
    const fetchResume = async () => {
      if (deviceId) {
        console.log("Device UUID (from state):", deviceId);

        const payload = {
          device_uuid: deviceId,
        };
        try {
          const response = await fetch("http://localhost:5000/api/get_resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          const result = await response.json();

          console.log(result);
          setResumeTokens(result.resume_tokens || []);
          setResumeText(result.resume_text || "");
        } catch (err) {
          console.error("Failed to fetch resume:", err);
        }
      }
    };

  fetchResume();
}, [deviceId]);

  if (!id) return <div>Please provide a job ID in the URL (e.g. /job?id=123)</div>;
  if (error) return <div>Error: {error}</div>;
  if (!job) return <div>Loading...</div>;

  return (
    <div className="job-viewer-container">
      <h1 className="job-viewer-title">{job?.title || "No Title Provided"}</h1>
  
      <p className="job-viewer-text">
        <span className="job-viewer-label">Company:</span> {job?.company_name || "N/A"}
      </p>
  
      <p className="job-viewer-text">
        <span className="job-viewer-label">Location:</span> {job?.location || "N/A"}
      </p>
  
      <p className="job-viewer-text">
        <span className="job-viewer-label">Description:</span>{" "}
        <span
          dangerouslySetInnerHTML={{
            __html: job?.description
              ? highlightDescriptionWithBoyerMoore(job.description, resumeTokens)
              : "<i>No description provided.</i>",
          }}
        />
      </p>
  
      <p className="job-viewer-text">
        <span className="job-viewer-label">Skills:</span>{" "}
        <span
          dangerouslySetInnerHTML={{
            __html: job?.skills_desc
              ? highlightDescriptionWithBoyerMoore(job.skills_desc, resumeTokens)
              : "<i>No skills listed.</i>",
          }}
        />
      </p>
  
      <p className="job-viewer-text">
        <span className="job-viewer-label">Work Type:</span> {job?.work_type || "N/A"}
      </p>
  
      <p className="job-viewer-text">
        <span className="job-viewer-label">Salary:</span>{" "}
        {job?.min_salary && job?.max_salary
          ? `${job.min_salary}–${job.max_salary} ${job?.currency || ""}`
          : "Not specified"}
      </p>
  
      <div className="ai-recommendation-box">
        <h2>🤖 AI Recommendations</h2>
  
        <details className="ai-section" onClick={() => handleExpand("job_fit")}>
          <summary>✅ Job Fit</summary>
          {loading.job_fit ? (
            <p>Loading...</p>
          ) : aiResponses.job_fit ? (
            <ReactMarkdown>{aiResponses.job_fit}</ReactMarkdown>
          ) : (
            <p>Click to generate recommendation.</p>
          )}
        </details>
  
        <details className="ai-section" onClick={() => handleExpand("tips")}>
          <summary>💡 Tips for Applying</summary>
          {loading.tips ? (
            <p>Loading...</p>
          ) : aiResponses.tips ? (
            <ReactMarkdown>{aiResponses.tips}</ReactMarkdown>
          ) : (
            <p>Click to generate recommendation.</p>
          )}
        </details>
  
        <details className="ai-section" onClick={() => handleExpand("skill_gap")}>
          <summary>🧠 Skill Gap and Recommended Courses</summary>
          {loading.skill_gap ? (
            <p>Loading...</p>
          ) : aiResponses.skill_gap ? (
            <ReactMarkdown>{aiResponses.skill_gap}</ReactMarkdown>
          ) : (
            <p>Click to generate recommendation.</p>
          )}
        </details>
  
        <details className="ai-section" onClick={() => handleExpand("legitimacy")}>
          <summary>🔍 Job Legitimacy</summary>
          {loading.legitimacy ? (
            <p>Loading...</p>
          ) : aiResponses.legitimacy ? (
            <ReactMarkdown>{aiResponses.legitimacy}</ReactMarkdown>
          ) : (
            <p>Click to generate recommendation.</p>
          )}
        </details>
      </div>
    </div>
  );
}

export default JobViewer;
