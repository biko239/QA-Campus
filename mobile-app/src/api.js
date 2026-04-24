const API_BASE = process.env.EXPO_PUBLIC_API_BASE_URL || "http://localhost:5110";

let sessionCookie = "";

async function request(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const headers = {
    ...(isForm ? {} : { "Content-Type": "application/json" }),
    ...(sessionCookie ? { Cookie: sessionCookie } : {}),
    ...(options.headers || {}),
  };

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  const setCookie = response.headers.get("set-cookie");
  if (setCookie) {
    sessionCookie = setCookie.split(";")[0];
  }

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new Error(data?.message || "Request failed.");
  }

  return data;
}

export const api = {
  login: (payload) => request("/api/auth/login", { method: "POST", body: JSON.stringify(payload) }),
  logout: () => {
    sessionCookie = "";
    return request("/api/auth/logout", { method: "POST", body: JSON.stringify({}) });
  },
  registerStudent: (payload) =>
    request("/api/auth/students/register", { method: "POST", body: JSON.stringify(payload) }),
  registerAdmin: (payload) =>
    request("/api/auth/admins/register", { method: "POST", body: JSON.stringify(payload) }),
  publicDocuments: () => request("/api/student/documents"),
  ask: (question) => request("/api/student/ask", { method: "POST", body: JSON.stringify({ question }) }),
  dashboard: () => request("/api/admin/dashboard"),
  documents: () => request("/api/admin/documents"),
  documentDetails: (id) => request(`/api/admin/documents/${id}`),
  uploadDocument: (formData) => request("/api/admin/documents", { method: "POST", body: formData }),
};
