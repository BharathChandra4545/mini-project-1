"use client";

import { useEffect, useMemo, useState } from "react";

type Row = Record<string, string | number | null>;
type Page = "Overview Dashboard" | "Performance Analysis" | "Predict Learning Risk" | "Student Report" | "About Project";
const features = ["G1", "G2", "studytime", "absences", "failures", "age"];

function number(row: Row, key: string): number | null {
  const value = Number(row[key]);
  return Number.isFinite(value) ? value : null;
}

function parseCsv(text: string): Row[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("CSV input must contain a header and at least one row.");
  const delimiter = lines[0].includes(";") ? ";" : ",";
  const headers = lines[0].split(delimiter).map((value) => value.trim().replace(/^"|"$/g, ""));
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(delimiter).map((value) => value.trim().replace(/^"|"$/g, ""));
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? null]));
  });
}

export default function Home() {
  const [page, setPage] = useState<Page>("Overview Dashboard");
  const [rows, setRows] = useState<Row[]>([]);
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState<Row | null>(null);
  const [report, setReport] = useState<Row[] | null>(null);
  const [form, setForm] = useState<Record<string, string>>({ G1: "10", G2: "10", studytime: "2", absences: "0", failures: "0", age: "17" });

  useEffect(() => {
    fetch("/api/data").then(async (response) => {
      const body = await response.json();
      if (!response.ok) throw new Error(body.error);
      setRows(body.rows);
    }).catch((reason: Error) => setError(reason.message));
  }, []);

  const grades = useMemo(() => rows.map((row) => number(row, "G3")).filter((value): value is number => value !== null), [rows]);
  const average = grades.length ? grades.reduce((sum, value) => sum + value, 0) / grades.length : 0;
  const atRisk = grades.filter((value) => value < 10).length;

  async function upload(file: File) {
    try {
      setRows(parseCsv(await file.text()));
      setReport(null);
      setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to read CSV."); }
  }

  async function predict(inputRows: Row[]) {
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rows: inputRows }) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    return body.rows as Row[];
  }

  async function submitPrediction(event: React.FormEvent) {
    event.preventDefault();
    try { setPrediction((await predict([form]))[0]); setError(""); } catch (reason) { setError(reason instanceof Error ? reason.message : "Prediction failed."); }
  }

  async function generateReport() {
    try { setReport(await predict(rows)); setError(""); } catch (reason) { setError(reason instanceof Error ? reason.message : "Report generation failed."); }
  }

  function downloadReport() {
    if (!report) return;
    const headers = Object.keys(report[0] ?? {});
    const csv = [headers.join(","), ...report.map((row) => headers.map((header) => JSON.stringify(row[header] ?? "")).join(","))].join("\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    link.download = "student_learning_risk_report.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return <main className="shell">
    <header><h1>🎓 Student Performance & Learning-Risk Analytics</h1><p>Academic insights • Early-warning indicators • Machine-learning predictions</p></header>
    <div className="layout">
      <aside><h2>📚 Navigation</h2>{(["Overview Dashboard", "Performance Analysis", "Predict Learning Risk", "Student Report", "About Project"] as Page[]).map((item) => <button className={page === item ? "active" : ""} onClick={() => setPage(item)} key={item}>{item}</button>)}<hr /><label className="upload">Upload student CSV (optional)<input type="file" accept=".csv" onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])} /></label><small>Semicolon and comma delimiters are supported.</small></aside>
      <section className="content">{error && <div className="error">{error}</div>}{page === "Overview Dashboard" && <><h2>📊 Academic Overview</h2><div className="metrics"><Metric label="Total Students" value={rows.length.toLocaleString()} /><Metric label="Average Final Grade" value={grades.length ? `${average.toFixed(2)}/20` : "N/A"} /><Metric label="Grade Below 10" value={atRisk.toString()} /><Metric label="Below-10 Percentage" value={grades.length ? `${(atRisk / grades.length * 100).toFixed(1)}%` : "N/A"} /></div><h3>Final Grade Distribution</h3><Bars values={grades} /><Table rows={rows.slice(0, 10)} /></>}{page === "Performance Analysis" && <><h2>📈 Performance Analysis</h2><div className="metrics"><Metric label="Highest Grade" value={grades.length ? `${Math.max(...grades)}/20` : "N/A"} /><Metric label="Lowest Grade" value={grades.length ? `${Math.min(...grades)}/20` : "N/A"} /><Metric label="Median Grade" value={grades.length ? `${[...grades].sort((a, b) => a - b)[Math.floor(grades.length / 2)]}/20` : "N/A"} /></div><h3>Grades Across Periods</h3><Table rows={rows.slice(0, 10).map((row) => ({ G1: row.G1, G2: row.G2, G3: row.G3, studytime: row.studytime, absences: row.absences }))} /></>}{page === "Predict Learning Risk" && <><h2>🧠 Individual Student Risk Prediction</h2><p>Estimate whether a student's final grade may fall below 10/20 using the trained Logistic Regression pipeline.</p><form onSubmit={submitPrediction} className="form">{features.map((key) => <label key={key}>{key}<input type="number" min={key === "age" ? 15 : 0} max={key === "G1" || key === "G2" ? 20 : undefined} value={form[key]} onChange={(event) => setForm({ ...form, [key]: event.target.value })} required /></label>)}<button className="primary" type="submit">Predict Learning Risk</button></form>{prediction && <div className="result"><Metric label="Estimated Risk Probability" value={`${(Number(prediction.Risk_Probability) * 100).toFixed(1)}%`} /><Metric label="Model Classification" value={Number(prediction.Predicted_Risk) ? "Flagged for review" : "Not flagged"} /></div>}</>}{page === "Student Report" && <><h2>📄 Student Report</h2><button className="primary" onClick={generateReport}>Generate Risk Report</button>{report && <><div className="metrics"><Metric label="Students Analyzed" value={report.length.toString()} /><Metric label="Flagged for Review" value={report.filter((row) => Number(row.Predicted_Risk) === 1).length.toString()} /></div><Table rows={report.slice(0, 50)} /><button onClick={downloadReport}>Download Student Risk Report (CSV)</button></>}</>}{page === "About Project" && <><h2>ℹ️ About This Project</h2><p>Analyze academic performance and explore early indicators that may help identify students who could benefit from additional support.</p><h3>Technologies used</h3><p>Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit, Next.js, and Vercel Python Functions.</p><p>The target is a final grade below 10 out of 20. Predictions are estimates and should not be used as the sole basis for educational decisions.</p></>}</section>
    </div>
  </main>;
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="metric"><small>{label}</small><strong>{value}</strong></div>; }
function Bars({ values }: { values: number[] }) { const counts = Array.from({ length: 21 }, (_, grade) => ({ grade, count: values.filter((value) => Math.round(value) === grade).length })); const max = Math.max(...counts.map((item) => item.count), 1); return <div className="bars">{counts.map((item) => <div className="bar" key={item.grade}><i style={{ height: `${item.count / max * 100}%` }} /><small>{item.grade}</small></div>)}</div>; }
function Table({ rows }: { rows: Row[] }) { if (!rows.length) return <p>No records available.</p>; const headers = Object.keys(rows[0]); return <div className="table-wrap"><table><thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{headers.map((header) => <td key={header}>{String(row[header] ?? "")}</td>)}</tr>)}</tbody></table></div>; }
