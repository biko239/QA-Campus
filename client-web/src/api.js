const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5110";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new Error(data?.message || "Request failed.");
  }

  return data;
}

export const api = {
  me: () => request("/api/auth/me"),
  login: (payload) => request("/api/auth/login", { method: "POST", body: JSON.stringify(payload) }),
  logout: () => request("/api/auth/logout", { method: "POST", body: JSON.stringify({}) }),
  registerStudent: (payload) =>
    request("/api/auth/students/register", { method: "POST", body: JSON.stringify(payload) }),
  registerAdmin: (payload) =>
    request("/api/auth/admins/register", { method: "POST", body: JSON.stringify(payload) }),
  publicDocuments: () => request("/api/student/documents"),
  chatHistory: () => request("/api/student/chat/history"),
  ask: (question) => request("/api/student/ask", { method: "POST", body: JSON.stringify({ question }) }),
  dashboard: () => request("/api/admin/dashboard"),
  documents: () => request("/api/admin/documents"),
  documentDetails: (id) => request(`/api/admin/documents/${id}`),
  uploadDocument: (formData) => request("/api/admin/documents", { method: "POST", body: formData }),
};
