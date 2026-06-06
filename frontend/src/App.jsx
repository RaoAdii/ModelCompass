import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import SummaryComparison from "./components/SummaryComparison";
import AdvancedMetricsDashboard from "./pages/AdvancedMetricsDashboard";
import AnalyticsDashboard from "./pages/AnalyticsDashboard";

function App() {
  return (
    <div>
      <header
        style={{
          padding: "0.9rem 1.2rem",
          borderBottom: "1px solid #d8e3e7",
          background: "#ffffff"
        }}
      >
        <nav style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
          <strong style={{ color: "#102a43" }}>ModelCompass</strong>
          <NavLink to="/" style={({ isActive }) => ({ color: isActive ? "#005f73" : "#5f6f7a" })}>
            Summarize
          </NavLink>
          <NavLink
            to="/analytics"
            style={({ isActive }) => ({ color: isActive ? "#005f73" : "#5f6f7a" })}
          >
            Analytics
          </NavLink>
          <NavLink
            to="/advanced"
            style={({ isActive }) => ({ color: isActive ? "#005f73" : "#5f6f7a" })}
          >
            Advanced Metrics
          </NavLink>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<SummaryComparison />} />
        <Route path="/analytics" element={<AnalyticsDashboard />} />
        <Route path="/advanced" element={<AdvancedMetricsDashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default App;
