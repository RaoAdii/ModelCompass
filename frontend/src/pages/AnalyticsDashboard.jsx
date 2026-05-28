import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

const palette = ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00"];

const cardStyle = {
  background: "rgba(255, 255, 255, 0.92)",
  border: "1px solid #d8e3e7",
  borderRadius: "12px",
  padding: "1rem"
};

function DashboardCard({ title, value, subtitle }) {
  return (
    <div style={cardStyle}>
      <p style={{ margin: "0 0 0.25rem", color: "#5f6f7a", fontSize: "0.9rem" }}>{title}</p>
      <h3 style={{ margin: "0 0 0.2rem", color: "#102a43", fontSize: "1.5rem" }}>{value}</h3>
      <p style={{ margin: 0, color: "#6f7f89", fontSize: "0.82rem" }}>{subtitle}</p>
    </div>
  );
}

export default function AnalyticsDashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v2/analytics`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Failed to load analytics data.");
        }
        setAnalytics(payload);
      } catch (requestError) {
        setError(requestError.message || "Unexpected error while loading analytics.");
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  const modelData = useMemo(() => {
    if (!analytics?.by_model) {
      return [];
    }
    return Object.entries(analytics.by_model).map(([model, metrics]) => ({
      model: model.toUpperCase(),
      avgRouge1: metrics.avg_rouge_1 || 0,
      avgLatency: metrics.avg_latency_ms || 0,
      count: metrics.count || 0
    }));
  }, [analytics]);

  const docTypeData = useMemo(() => {
    if (!analytics?.by_doc_type) {
      return [];
    }
    return Object.entries(analytics.by_doc_type).map(([docType, metrics]) => ({
      docType,
      count: metrics.count || 0,
      avgRouge1: metrics.avg_rouge_1 || 0
    }));
  }, [analytics]);

  const latencyTrendData = useMemo(() => {
    if (!analytics?.recent_summaries) {
      return [];
    }
    return [...analytics.recent_summaries]
      .reverse()
      .map((row, index) => ({
        seq: index + 1,
        latency: row.latency_ms || 0,
        rouge1: row.rouge_1 || 0
      }));
  }, [analytics]);

  if (loading) {
    return (
      <div style={{ padding: "2rem", color: "#102a43" }}>
        <h2 style={{ margin: 0 }}>Analytics Dashboard</h2>
        <p style={{ marginTop: "0.6rem" }}>Loading analytics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: "2rem", color: "#b00020" }}>
        <h2 style={{ margin: 0 }}>Analytics Dashboard</h2>
        <p style={{ marginTop: "0.6rem" }}>{error}</p>
      </div>
    );
  }

  const summary = analytics?.summary || {};
  const recentRows = analytics?.recent_summaries || [];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "radial-gradient(circle at top right, #d7eef0, #f3f6f7 60%)",
        padding: "1.2rem"
      }}
    >
      <div style={{ maxWidth: "1200px", margin: "0 auto", display: "grid", gap: "1rem" }}>
        <div style={cardStyle}>
          <h1 style={{ margin: 0, color: "#102a43" }}>Analytics Dashboard</h1>
          <p style={{ margin: "0.4rem 0 0", color: "#5f6f7a" }}>
            Track cache utilization, model performance, and summarization latency.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))"
          }}
        >
          <DashboardCard
            title="Total Documents"
            value={Number(summary.total_documents || 0).toLocaleString()}
            subtitle="Tracked by /api/v2 summarize requests"
          />
          <DashboardCard
            title="Cache Hit Rate"
            value={`${((summary.cache_hit_rate || 0) * 100).toFixed(1)}%`}
            subtitle="Higher means faster repeat requests"
          />
          <DashboardCard
            title="Average Latency"
            value={`${Number(summary.avg_inference_time_ms || 0).toFixed(0)} ms`}
            subtitle="End-to-end inference time"
          />
        </div>

        <div
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))"
          }}
        >
          <div style={cardStyle}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Model Performance (ROUGE-1)</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={modelData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="avgRouge1" name="Avg ROUGE-1">
                  {modelData.map((_, index) => (
                    <Cell key={`m-${index}`} fill={palette[index % palette.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={cardStyle}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Document Type Distribution</h3>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={docTypeData} dataKey="count" nameKey="docType" cx="50%" cy="50%" outerRadius={90} label>
                  {docTypeData.map((_, index) => (
                    <Cell key={`d-${index}`} fill={palette[index % palette.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ marginTop: 0, color: "#102a43" }}>Inference Latency Trend</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={latencyTrendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="seq" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="latency" stroke="#005f73" name="Latency (ms)" strokeWidth={2} />
              <Line type="monotone" dataKey="rouge1" stroke="#ee9b00" name="ROUGE-1" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={cardStyle}>
          <h3 style={{ marginTop: 0, color: "#102a43" }}>Recent Summaries</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "700px" }}>
              <thead>
                <tr style={{ background: "#edf4f6" }}>
                  <th style={{ textAlign: "left", padding: "0.6rem" }}>Timestamp</th>
                  <th style={{ textAlign: "left", padding: "0.6rem" }}>Doc Type</th>
                  <th style={{ textAlign: "left", padding: "0.6rem" }}>Model</th>
                  <th style={{ textAlign: "left", padding: "0.6rem" }}>ROUGE-1</th>
                  <th style={{ textAlign: "left", padding: "0.6rem" }}>Latency (ms)</th>
                  <th style={{ textAlign: "left", padding: "0.6rem" }}>Cached</th>
                </tr>
              </thead>
              <tbody>
                {recentRows.length === 0 && (
                  <tr>
                    <td colSpan={6} style={{ padding: "0.8rem", color: "#5f6f7a" }}>
                      No records available yet.
                    </td>
                  </tr>
                )}
                {recentRows.map((row, index) => (
                  <tr key={`${row.timestamp}-${index}`} style={{ borderTop: "1px solid #e6ecef" }}>
                    <td style={{ padding: "0.6rem" }}>{new Date(row.timestamp).toLocaleString()}</td>
                    <td style={{ padding: "0.6rem" }}>{row.doc_type}</td>
                    <td style={{ padding: "0.6rem" }}>{String(row.model || "").toUpperCase()}</td>
                    <td style={{ padding: "0.6rem" }}>{Number(row.rouge_1 || 0).toFixed(3)}</td>
                    <td style={{ padding: "0.6rem" }}>{Number(row.latency_ms || 0).toFixed(0)}</td>
                    <td style={{ padding: "0.6rem" }}>{row.was_cached ? "Yes" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

