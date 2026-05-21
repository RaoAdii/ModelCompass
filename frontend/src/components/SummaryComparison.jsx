import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import "./SummaryComparison.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

function SummaryCard({ title, summary }) {
  const copySummary = async () => {
    await navigator.clipboard.writeText(summary);
  };

  return (
    <div className="summary-card">
      <div className="summary-header">
        <h3>{title}</h3>
        <button type="button" onClick={copySummary}>
          Copy
        </button>
      </div>
      <p>{summary}</p>
    </div>
  );
}

function SummaryComparison() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [documentType, setDocumentType] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const rougeData = useMemo(() => {
    if (!result?.evaluation?.pairwise_rouge) {
      return [];
    }

    return Object.entries(result.evaluation.pairwise_rouge).map(([pair, scores]) => ({
      pair,
      rouge1: scores.rouge1,
      rouge2: scores.rouge2,
      rougeL: scores.rougeL
    }));
  }, [result]);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setError("");
  };

  const onFileInputChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const submitFile = async () => {
    if (!selectedFile) {
      setError("Please upload a PDF or TXT file.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (documentType) {
      formData.append("document_type", documentType);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/summarize`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Unable to summarize document.");
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "Unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <h1>ModelCompass</h1>
        <p>Upload a document to compare BART, Pegasus, and T5 summaries.</p>

        <div
          className={`drop-zone ${isDragging ? "dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
        >
          <input type="file" accept=".pdf,.txt" onChange={onFileInputChange} />
          <span>{selectedFile ? selectedFile.name : "Drag and drop a file here or click to browse"}</span>
        </div>

        <div className="controls">
          <select value={documentType} onChange={(event) => setDocumentType(event.target.value)}>
            <option value="">Auto detect document type</option>
            <option value="research_paper">Research Paper</option>
            <option value="announcement">Announcement</option>
            <option value="news">News</option>
          </select>
          <button type="button" onClick={submitFile} disabled={isLoading}>
            {isLoading ? "Summarizing..." : "Generate Summaries"}
          </button>
        </div>

        {isLoading && <div className="spinner" aria-label="Loading spinner" />}
        {error && <p className="error">{error}</p>}

        {result && (
          <div className="results">
            <p className="meta">
              Detected Type: <strong>{result.document_type}</strong> | Routed Model:{" "}
              <strong>{result.routed_model}</strong>
            </p>
            <div className="summary-grid">
              <SummaryCard title="BART" summary={result.summaries.summary_bart} />
              <SummaryCard title="Pegasus" summary={result.summaries.summary_pegasus} />
              <SummaryCard title="T5" summary={result.summaries.summary_t5} />
            </div>
            <div className="chart-container">
              <h2>Pairwise ROUGE Comparison</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={rougeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="pair" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="rouge1" fill="#005f73" />
                  <Bar dataKey="rouge2" fill="#0a9396" />
                  <Bar dataKey="rougeL" fill="#94d2bd" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SummaryComparison;
