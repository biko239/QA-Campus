import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, FileText, LayoutDashboard, LogOut, MessageSquare, Upload } from "lucide-react";
import { api } from "./api";
import "./styles.css";

const emptyLogin = { identifier: "", password: "", accessCode: "" };

function App() {
  const [session, setSession] = useState(null);
  const [view, setView] = useState("login");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    api.me()
      .then((me) => {
        setSession(me);
        setView(me.role === "Admin" ? "adminDashboard" : "chat");
      })
      .catch(() => {});
  }, []);

  async function logout() {
    await api.logout();
    setSession(null);
    setNotice("");
    setView("login");
  }

  const area = session?.role === "Admin" ? "Administrator Area" : "Student Area";

  return (
    <div className="portal-body">
      <header className="top-header">
        <div className="top-header-inner">
          <div className="brand-block">
            <div className="brand-title">University Portal</div>
            <div className="brand-subtitle">{area}</div>
          </div>

          <nav className="top-nav">
            {!session && <button onClick={() => setView("login")}>Login</button>}
            {!session && <button onClick={() => setView("studentRegister")}>Register</button>}
            {session?.role === "Student" && <button onClick={() => setView("chat")}>Student Chat</button>}
            {session?.role === "Admin" && <button onClick={() => setView("adminDashboard")}>Dashboard</button>}
            {session?.role === "Admin" && <button onClick={() => setView("upload")}>Upload Documents</button>}
            {session?.role === "Admin" && <button onClick={() => setView("documents")}>Documents</button>}
          </nav>

          <div className="header-actions">
            <span className="user-badge">{session?.name || "Guest"}</span>
            {session && (
              <button className="logout-btn" onClick={logout} title="Logout">
                <LogOut size={16} />
                Logout
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="page-shell">
        {notice && <div className="success-box">{notice}</div>}
        {view === "login" && <Login setSession={setSession} setView={setView} setNotice={setNotice} />}
        {view === "studentRegister" && <StudentRegister setView={setView} setNotice={setNotice} />}
        {view === "adminRegister" && <AdminRegister setView={setView} setNotice={setNotice} />}
        {view === "chat" && <StudentChat />}
        {view === "adminDashboard" && <AdminDashboard setView={setView} />}
        {view === "upload" && <UploadDocument setView={setView} setNotice={setNotice} />}
        {view === "documents" && <Documents setView={setView} />}
        {view.startsWith("document:") && <DocumentDetails id={view.split(":")[1]} setView={setView} />}
      </main>
    </div>
  );
}

function Login({ setSession, setView, setNotice }) {
  const [form, setForm] = useState(emptyLogin);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setError("");
    setNotice("");

    try {
      const nextSession = await api.login({
        identifier: form.identifier,
        password: form.password,
        accessCode: form.accessCode || null,
      });
      setSession(nextSession);
      setView(nextSession.role === "Admin" ? "adminDashboard" : "chat");
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section className="auth-card">
      <h1>University Portal</h1>
      <p className="subtitle">Unified Login</p>
      <hr />
      {error && <div className="error-box">{error}</div>}
      <form onSubmit={submit}>
        <Field label="Email or Username" value={form.identifier} onChange={(identifier) => setForm({ ...form, identifier })} />
        <Field label="Password" type="password" value={form.password} onChange={(password) => setForm({ ...form, password })} />
        <Field label="Access Code (admins only)" type="password" value={form.accessCode} onChange={(accessCode) => setForm({ ...form, accessCode })} />
        <div className="form-actions-column">
          <button className="primary-btn wide-btn">Login</button>
          <hr />
          <p className="center-text">Don't have a student account?</p>
          <button type="button" className="link-btn" onClick={() => setView("studentRegister")}>Create Student Account</button>
          <p className="center-text">Need an admin account?</p>
          <button type="button" className="link-btn" onClick={() => setView("adminRegister")}>Create Admin Account</button>
        </div>
      </form>
    </section>
  );
}

function StudentRegister({ setView, setNotice }) {
  const [form, setForm] = useState({ firstName: "", lastName: "", studentNumber: "", email: "", password: "", confirmPassword: "", department: "", termsAccepted: false });
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setError("");
    try {
      const result = await api.registerStudent(form);
      setNotice(result.message);
      setView("login");
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section className="auth-card large">
      <h1>University Portal</h1>
      <p className="subtitle">Student Registration</p>
      <hr />
      {error && <div className="error-box">{error}</div>}
      <form className="grid-form" onSubmit={submit}>
        <Field label="First Name" value={form.firstName} onChange={(firstName) => setForm({ ...form, firstName })} />
        <Field label="Last Name" value={form.lastName} onChange={(lastName) => setForm({ ...form, lastName })} />
        <Field className="full" label="Student ID" value={form.studentNumber} onChange={(studentNumber) => setForm({ ...form, studentNumber })} />
        <Field className="full" label="Email Address" type="email" value={form.email} onChange={(email) => setForm({ ...form, email })} />
        <Field label="Password" type="password" value={form.password} onChange={(password) => setForm({ ...form, password })} />
        <Field label="Confirm Password" type="password" value={form.confirmPassword} onChange={(confirmPassword) => setForm({ ...form, confirmPassword })} />
        <div className="full">
          <label>Department</label>
          <select value={form.department} onChange={(e) => setForm({ ...form, department: e.target.value })}>
            <option value="">Select Department</option>
            <option>Computer Science</option>
            <option>Business</option>
            <option>Engineering</option>
            <option>Medicine</option>
          </select>
        </div>
        <label className="full checkbox-row">
          <input type="checkbox" checked={form.termsAccepted} onChange={(e) => setForm({ ...form, termsAccepted: e.target.checked })} />
          I agree to the terms and conditions
        </label>
        <div className="full form-actions-column">
          <button className="primary-btn wide-btn">Create Account</button>
          <button type="button" className="link-btn" onClick={() => setView("login")}>Login</button>
        </div>
      </form>
    </section>
  );
}

function AdminRegister({ setView, setNotice }) {
  const [form, setForm] = useState({ username: "", password: "", confirmPassword: "", accessCode: "", confirmAccessCode: "" });
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    try {
      const result = await api.registerAdmin(form);
      setNotice(result.message);
      setView("login");
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section className="auth-card">
      <h1>University Portal</h1>
      <p className="subtitle">Create Admin Account</p>
      <hr />
      {error && <div className="error-box">{error}</div>}
      <form onSubmit={submit}>
        {["username", "password", "confirmPassword", "accessCode", "confirmAccessCode"].map((key) => (
          <Field key={key} label={labelize(key)} type={key.toLowerCase().includes("password") || key.toLowerCase().includes("code") ? "password" : "text"} value={form[key]} onChange={(value) => setForm({ ...form, [key]: value })} />
        ))}
        <button className="primary-btn wide-btn">Create Admin Account</button>
        <button type="button" className="link-btn" onClick={() => setView("login")}>Back to Login</button>
      </form>
    </section>
  );
}

function StudentChat() {
  const [docs, setDocs] = useState([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.publicDocuments().then(setDocs).catch((err) => setError(err.message));
  }, []);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await api.ask(question);
      setAnswer(result.answer);
      setCitations(result.citations || []);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageTitle title="Q&A Chat Interface" subtitle="Ask questions about your university documents" icon={<MessageSquare />} />
      {error && <div className="error-box">{error}</div>}
      <div className="chat-shell">
        <aside className="chat-sidebar">
          <button className="primary-btn wide-btn" onClick={() => { setAnswer(""); setCitations([]); }}>New Chat</button>
          <div className="docs-panel">
            <h3>Available Documents</h3>
            {docs.length ? docs.map((doc) => <p key={doc.id}><strong>{doc.title}</strong><br /><span>{doc.department} - {doc.category}</span></p>) : <p>No processed public documents available yet.</p>}
          </div>
        </aside>
        <section className="chat-main">
          <div className="chat-messages">
            {answer ? <div className="docs-panel"><strong>Answer</strong><p className="pre-line">{answer}</p></div> : <p>Welcome. Ask a question and the assistant will answer using the uploaded documents.</p>}
            {citations.length > 0 && <div className="docs-panel"><h3>Proof / Retrieved Chunks</h3>{citations.map((citation) => <div className="citation" key={citation.chunkId}><strong>{citation.documentTitle}</strong><br /><span>Chunk ID: {citation.chunkId} | Score: {citation.score}</span><p>{citation.preview}</p></div>)}</div>}
          </div>
          <form className="chat-input-bar" onSubmit={submit}>
            <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Type your question here..." />
            <button className="primary-btn" disabled={busy}>{busy ? "Sending..." : "Send"}</button>
          </form>
        </section>
      </div>
    </>
  );
}

function AdminDashboard({ setView }) {
  const [stats, setStats] = useState(null);
  useEffect(() => { api.dashboard().then(setStats); }, []);
  const values = stats || { students: 0, documents: 0, questions: 0, answers: 0 };

  return (
    <>
      <PageTitle title="Admin Dashboard" subtitle="University Portal Management" icon={<LayoutDashboard />} />
      <div className="dashboard-grid">{Object.entries(values).map(([key, value]) => <div className="metric-card" key={key}><h3>{labelize(key)}</h3><div className="metric-value">{value}</div></div>)}</div>
      <div className="split-grid">
        <section className="section-card"><h2>Recent Activity</h2><ul><li>Student registration</li><li>Document uploaded</li><li>Question answered</li></ul></section>
        <section className="section-card"><h2>Quick Actions</h2><div className="action-list"><button className="primary-link" onClick={() => setView("upload")}>Upload Documents</button><button onClick={() => setView("documents")}>Document Library</button></div></section>
      </div>
    </>
  );
}

function UploadDocument({ setView, setNotice }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const data = new FormData(e.currentTarget);
    data.set("isPublic", data.get("isPublic") === "on" ? "true" : "false");
    try {
      const result = await api.uploadDocument(data);
      setNotice(result.message);
      setView("documents");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageTitle title="Document Upload" subtitle="Upload course materials and resources" icon={<Upload />} />
      <section className="auth-card large">
        {error && <div className="error-box">{error}</div>}
        <form className="grid-form" onSubmit={submit}>
          <Field className="full" label="Select File" type="file" name="file" />
          <Field className="full" label="Document Title" name="title" />
          <Field label="Department" name="department" />
          <Field label="Course" name="course" />
          <Field label="Category" name="category" />
          <div className="full"><label>Description</label><textarea name="description" /></div>
          <Field className="full" label="Tags" name="tags" />
          <label className="full checkbox-row"><input type="checkbox" name="isPublic" /> Make this document publicly available</label>
          <button className="full primary-btn wide-btn" disabled={busy}>{busy ? "Processing..." : "Upload Document"}</button>
        </form>
      </section>
    </>
  );
}

function Documents({ setView }) {
  const [docs, setDocs] = useState([]);
  useEffect(() => { api.documents().then(setDocs); }, []);
  return (
    <>
      <PageTitle title="Document Library" subtitle="Browse and manage course documents" icon={<BookOpen />} />
      <div className="table-panel">
        <table className="portal-table">
          <thead><tr><th>Document Name</th><th>Department</th><th>Category</th><th>Upload Date</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>{docs.map((doc) => <tr key={doc.id}><td>{doc.title}</td><td>{doc.department}</td><td>{doc.category}</td><td>{new Date(doc.uploadedAt).toLocaleString()}</td><td>{doc.status}</td><td><button className="table-link" onClick={() => setView(`document:${doc.id}`)}>Details</button></td></tr>)}</tbody>
        </table>
      </div>
    </>
  );
}

function DocumentDetails({ id, setView }) {
  const [doc, setDoc] = useState(null);
  useEffect(() => { api.documentDetails(id).then(setDoc); }, [id]);
  const chunks = useMemo(() => doc?.chunks || [], [doc]);

  if (!doc) return <div className="section-card">Loading document...</div>;

  return (
    <>
      <PageTitle title="Document Details" subtitle="View document metadata and chunks" icon={<FileText />} />
      <section className="section-card">
        {["title", "department", "course", "category", "status", "fileType", "fileSize", "processingMethod"].map((key) => <p key={key}><strong>{labelize(key)}:</strong> {doc[key]}</p>)}
        <button className="secondary-btn" onClick={() => setView("documents")}>Back to List</button>
      </section>
      <section className="section-card">
        <h2>Chunks</h2>
        {chunks.length ? chunks.map((chunk) => <div className="docs-panel" key={chunk.id}><strong>Chunk #{chunk.chunkIndex}</strong><p className="pre-line">{chunk.text}</p></div>) : <p>No chunks found for this document.</p>}
      </section>
    </>
  );
}

function PageTitle({ title, subtitle, icon }) {
  return <div className="top-page-title"><div className="title-row">{icon}<h1>{title}</h1></div><p>{subtitle}</p></div>;
}

function Field({ label, value, onChange, type = "text", className = "", name }) {
  return <div className={className}><label>{label}</label><input name={name} type={type} value={value} onChange={onChange ? (e) => onChange(e.target.value) : undefined} /></div>;
}

function labelize(value) {
  return value.replace(/([A-Z])/g, " $1").replace(/^./, (letter) => letter.toUpperCase());
}

createRoot(document.getElementById("root")).render(<App />);
