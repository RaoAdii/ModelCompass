import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";
const COLORS = {
  bart: "#005f73",
  pegasus: "#0a9396",
  t5: "#ee9b00",
  neutral: "#5f6f7a"
};

const cardStyle = {
  background: "rgba(255, 255, 255, 0.94)",
  border: "1px solid #d8e3e7",
  borderRadius: "8px",
  padding: "1rem"
};

function normalizeModel(model) {
  return String(model || "").toLowerCase();
}

function MetricCard({ title, value, subtitle }) {
  return (
    <section style={cardStyle}>
      <p style={{ margin: "0 0 0.25rem", color: "#5f6f7a", fontSize: "0.88rem" }}>{title}</p>
      <h2 style={{ margin: 0, color: "#102a43", fontSize: "1.45rem" }}>{value}</h2>
      <p style={{ margin: "0.25rem 0 0", color: "#6f7f89", fontSize: "0.82rem" }}>{subtitle}</p>
    </section>
  );
}

export default function AdvancedMetricsDashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [advanced, setAdvanced] = useState(null);
  const [sortKey, setSortKey] = useState("rouge_1");
  const [page, setPage] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [analyticsResponse, advancedResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/api/v2/analytics`),
          fetch(`${API_BASE_URL}/api/v2/analytics/advanced`)
        ]);
        const [analyticsPayload, advancedPayload] = await Promise.all([
          analyticsResponse.json(),
          advancedResponse.json()
        ]);
        if (!analyticsResponse.ok) {
          throw new Error(analyticsPayload.error || "Unable to load analytics.");
        }
        if (!advancedResponse.ok) {
          throw new Error(advancedPayload.error || "Unable to load advanced analytics.");
        }
        setAnalytics(analyticsPayload);
        setAdvanced(advancedPayload);
      } catch (requestError) {
        setError(requestError.message || "Unexpected error while loading metrics.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const rows = useMemo(() => {
    const recent = analytics?.recent_summaries || [];
    return recent.map((item, index) => ({
      id: `${item.timestamp}-${index}`,
      timestamp: item.timestamp,
      doc_type: item.doc_type,
      model: normalizeModel(item.model),
      rouge_1: Number(item.rouge_1 || 0),
      semantic_similarity: Number(item.semantic_similarity || 0),
      abstractiveness: Number(item.abstractiveness || 0),
      entity_preservation: Number(item.entity_preservation || 0),
      latency_ms: Number(item.latency_ms || 0)
    }));
  }, [analytics]);

  const sortedRows = useMemo(() => {
    return [...rows].sort((left, right) => {
      const leftValue = left[sortKey];
      const rightValue = right[sortKey];
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return rightValue - leftValue;
      }
      return String(leftValue).localeCompare(String(rightValue));
    });
  }, [rows, sortKey]);

  const pageRows = sortedRows.slice(page * 10, page * 10 + 10);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / 10));

  const radarData = useMemo(() => {
    const byModel = analytics?.by_model || {};
    const advancedAvg = advanced?.advanced_metrics_avg || {};
    return ["bart", "pegasus", "t5"].map((model) => ({
      metric: model.toUpperCase(),
      rouge1: byModel[model]?.avg_rouge_1 || 0,
      rougeL: byModel[model]?.avg_rouge_1 || 0,
      semantic: advancedAvg[model]?.avg_semantic_similarity || 0,
      abstractiveness: advancedAvg[model]?.avg_abstractiveness || 0,
      entity: advancedAvg[model]?.avg_entity_preservation || 0
    }));
  }, [analytics, advanced]);

  const scatterData = useMemo(() => {
    return ["bart", "pegasus", "t5"].map((model) => ({
      model,
      data: rows.filter((row) => row.model === model)
    }));
  }, [rows]);

  const docTypePie = useMemo(() => {
    return Object.entries(analytics?.by_doc_type || {}).map(([name, metrics]) => ({
      name,
      value: metrics.count || 0
    }));
  }, [analytics]);

  if (loading) {
    return <main style={{ padding: "2rem", color: "#102a43" }}>Loading advanced metrics...</main>;
  }

  if (error) {
    return <main style={{ padding: "2rem", color: "#b00020" }}>{error}</main>;
  }

  const summary = analytics?.summary || {};
  const classifier = advanced?.classifier_accuracy || {};
  const lastResearch = rows.find((row) => row.doc_type === "research_paper");

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #f3f6f7 0%, #e6f0ef 55%, #f7f1df 100%)",
        padding: "1.2rem"
      }}
    >
      <div style={{ maxWidth: "1240px", margin: "0 auto", display: "grid", gap: "1rem" }}>
        <section style={{ ...cardStyle, display: "grid", gap: "0.35rem" }}>
          <h1 style={{ margin: 0, color: "#102a43", fontSize: "1.8rem" }}>Advanced Metrics</h1>
          <p style={{ margin: 0, color: "#5f6f7a" }}>
            Semantic fidelity, abstractiveness, entity preservation, and classifier behavior.
          </p>
        </section>

        <section
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))"
          }}
        >
          <MetricCard
            title="Documents"
            value={Number(summary.total_documents || 0).toLocaleString()}
            subtitle="v2 analytics rows"
          />
          <MetricCard
            title="Classifier"
            value={classifier.method || "tfidf"}
            subtitle="Current dominant method"
          />
          <MetricCard
            title="Cache Hit Rate"
            value={`${((summary.cache_hit_rate || 0) * 100).toFixed(1)}%`}
            subtitle="Repeat document reuse"
          />
        </section>

        <section
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))"
          }}
        >
          <div style={cardStyle}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Model Strength Radar</h3>
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis domain={[0, 1]} />
                <Radar dataKey="rouge1" name="ROUGE-1" stroke="#005f73" fill="#005f73" fillOpacity={0.18} />
                <Radar dataKey="semantic" name="Semantic" stroke="#0a9396" fill="#0a9396" fillOpacity={0.16} />
                <Radar dataKey="entity" name="Entity" stroke="#ee9b00" fill="#ee9b00" fillOpacity={0.14} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div style={cardStyle}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Quality vs Abstractiveness</h3>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="abstractiveness" name="Abstractiveness" domain={[0, 1]} />
                <YAxis dataKey="semantic_similarity" name="Semantic" domain={[0, 1]} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                {scatterData.map((group) => (
                  <Scatter
                    key={group.model}
                    name={group.model.toUpperCase()}
                    data={group.data}
                    fill={COLORS[group.model]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "minmax(280px, 1fr) minmax(280px, 1fr)"
          }}
        >
          <div style={cardStyle}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Document Mix</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={docTypePie} dataKey="value" nameKey="name" outerRadius={85} label>
                  {docTypePie.map((_, index) => (
                    <Cell
                      key={`doc-${index}`}
                      fill={[COLORS.bart, COLORS.pegasus, COLORS.t5][index % 3]}
                    />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div style={cardStyle}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Citation Panel</h3>
            {lastResearch ? (
              <div style={{ color: "#5f6f7a", lineHeight: 1.7 }}>
                <p style={{ margin: 0 }}>Last research-paper row: {new Date(lastResearch.timestamp).toLocaleString()}</p>
                <p style={{ margin: 0 }}>Model: {lastResearch.model.toUpperCase()}</p>
                <p style={{ margin: 0 }}>Entity preservation: {lastResearch.entity_preservation.toFixed(3)}</p>
              </div>
            ) : (
              <p style={{ margin: 0, color: "#5f6f7a" }}>No research-paper analytics are available yet.</p>
            )}
          </div>
        </section>

        <section style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
            <h3 style={{ marginTop: 0, color: "#102a43" }}>Metrics Table</h3>
            <select
              value={sortKey}
              onChange={(event) => {
                setSortKey(event.target.value);
                setPage(0);
              }}
              style={{ border: "1px solid #b6c8cf", borderRadius: "8px", padding: "0.45rem" }}
            >
              <option value="rouge_1">ROUGE-1</option>
              <option value="semantic_similarity">Semantic similarity</option>
              <option value="abstractiveness">Abstractiveness</option>
              <option value="entity_preservation">Entity preservation</option>
              <option value="latency_ms">Latency</option>
            </select>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "820px" }}>
              <thead>
                <tr style={{ background: "#edf4f6" }}>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>Doc Type</th>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>Model</th>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>ROUGE-1</th>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>Semantic</th>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>Abstractiveness</th>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>Entity</th>
                  <th style={{ textAlign: "left", padding: "0.65rem" }}>Latency</th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row) => (
                  <tr key={row.id} style={{ borderTop: "1px solid #e6ecef" }}>
                    <td style={{ padding: "0.65rem" }}>{row.doc_type}</td>
                    <td style={{ padding: "0.65rem" }}>{row.model.toUpperCase()}</td>
                    <td style={{ padding: "0.65rem" }}>{row.rouge_1.toFixed(3)}</td>
                    <td style={{ padding: "0.65rem" }}>{row.semantic_similarity.toFixed(3)}</td>
                    <td style={{ padding: "0.65rem" }}>{row.abstractiveness.toFixed(3)}</td>
                    <td style={{ padding: "0.65rem" }}>{row.entity_preservation.toFixed(3)}</td>
                    <td style={{ padding: "0.65rem" }}>{row.latency_ms.toFixed(0)} ms</td>
                  </tr>
                ))}
                {pageRows.length === 0 && (
                  <tr>
                    <td colSpan={7} style={{ padding: "0.8rem", color: COLORS.neutral }}>
                      No advanced metric rows are available yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.6rem", marginTop: "0.8rem" }}>
            <button disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
              Previous
            </button>
            <span style={{ alignSelf: "center", color: COLORS.neutral }}>
              Page {page + 1} of {totalPages}
            </span>
            <button
              disabled={page + 1 >= totalPages}
              onClick={() => setPage((value) => Math.min(totalPages - 1, value + 1))}
            >
              Next
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}
