import React, { useEffect, useState } from "react";
import { Alert, ScrollView, StyleSheet, Switch, Text, TextInput, TouchableOpacity, View } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { api } from "./src/api";

export default function App() {
  const [session, setSession] = useState(null);
  const [screen, setScreen] = useState("login");
  const [notice, setNotice] = useState("");

  async function logout() {
    await api.logout().catch(() => {});
    setSession(null);
    setScreen("login");
  }

  return (
    <View style={styles.app}>
      <View style={styles.header}>
        <View>
          <Text style={styles.brand}>University Portal</Text>
          <Text style={styles.subtitle}>{session?.role === "Admin" ? "Administrator Area" : "Student Area"}</Text>
        </View>
        {session && <Button label="Logout" variant="dark" onPress={logout} />}
      </View>

      <ScrollView contentContainerStyle={styles.shell}>
        {notice ? <Text style={styles.success}>{notice}</Text> : null}
        {screen === "login" && <Login setSession={setSession} setScreen={setScreen} setNotice={setNotice} />}
        {screen === "studentRegister" && <StudentRegister setScreen={setScreen} setNotice={setNotice} />}
        {screen === "adminRegister" && <AdminRegister setScreen={setScreen} setNotice={setNotice} />}
        {screen === "chat" && <StudentChat />}
        {screen === "dashboard" && <AdminDashboard setScreen={setScreen} />}
        {screen === "upload" && <UploadDocument setScreen={setScreen} setNotice={setNotice} />}
        {screen === "documents" && <Documents setScreen={setScreen} />}
        {screen.startsWith("document:") && <DocumentDetails id={screen.split(":")[1]} setScreen={setScreen} />}
      </ScrollView>
    </View>
  );
}

function Login({ setSession, setScreen, setNotice }) {
  const [form, setForm] = useState({ identifier: "", password: "", accessCode: "" });

  async function submit() {
    try {
      setNotice("");
      const next = await api.login(form);
      setSession(next);
      setScreen(next.role === "Admin" ? "dashboard" : "chat");
    } catch (err) {
      Alert.alert("Login failed", err.message);
    }
  }

  return (
    <Card title="University Portal" subtitle="Unified Login">
      <Input label="Email or Username" value={form.identifier} onChangeText={(identifier) => setForm({ ...form, identifier })} />
      <Input label="Password" secureTextEntry value={form.password} onChangeText={(password) => setForm({ ...form, password })} />
      <Input label="Access Code (admins only)" secureTextEntry value={form.accessCode} onChangeText={(accessCode) => setForm({ ...form, accessCode })} />
      <Button label="Login" variant="dark" onPress={submit} />
      <Button label="Create Student Account" onPress={() => setScreen("studentRegister")} />
      <Button label="Create Admin Account" onPress={() => setScreen("adminRegister")} />
    </Card>
  );
}

function StudentRegister({ setScreen, setNotice }) {
  const [form, setForm] = useState({ firstName: "", lastName: "", studentNumber: "", email: "", department: "", password: "", confirmPassword: "", termsAccepted: false });

  async function submit() {
    try {
      const result = await api.registerStudent(form);
      setNotice(result.message);
      setScreen("login");
    } catch (err) {
      Alert.alert("Registration failed", err.message);
    }
  }

  return (
    <Card title="University Portal" subtitle="Student Registration">
      {["firstName", "lastName", "studentNumber", "email", "department"].map((key) => (
        <Input key={key} label={labelize(key)} value={form[key]} onChangeText={(value) => setForm({ ...form, [key]: value })} />
      ))}
      <Input label="Password" secureTextEntry value={form.password} onChangeText={(password) => setForm({ ...form, password })} />
      <Input label="Confirm Password" secureTextEntry value={form.confirmPassword} onChangeText={(confirmPassword) => setForm({ ...form, confirmPassword })} />
      <View style={styles.switchRow}><Switch value={form.termsAccepted} onValueChange={(termsAccepted) => setForm({ ...form, termsAccepted })} /><Text>I agree to the terms and conditions</Text></View>
      <Button label="Create Account" variant="dark" onPress={submit} />
      <Button label="Back to Login" onPress={() => setScreen("login")} />
    </Card>
  );
}

function AdminRegister({ setScreen, setNotice }) {
  const [form, setForm] = useState({ username: "", password: "", confirmPassword: "", accessCode: "", confirmAccessCode: "" });

  async function submit() {
    try {
      const result = await api.registerAdmin(form);
      setNotice(result.message);
      setScreen("login");
    } catch (err) {
      Alert.alert("Registration failed", err.message);
    }
  }

  return (
    <Card title="University Portal" subtitle="Create Admin Account">
      {Object.keys(form).map((key) => (
        <Input key={key} label={labelize(key)} secureTextEntry={key.toLowerCase().includes("password") || key.toLowerCase().includes("code")} value={form[key]} onChangeText={(value) => setForm({ ...form, [key]: value })} />
      ))}
      <Button label="Create Admin Account" variant="dark" onPress={submit} />
      <Button label="Back to Login" onPress={() => setScreen("login")} />
    </Card>
  );
}

function StudentChat() {
  const [docs, setDocs] = useState([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);

  useEffect(() => {
    api.publicDocuments().then(setDocs).catch((err) => Alert.alert("Documents", err.message));
  }, []);

  async function ask() {
    try {
      const result = await api.ask(question);
      setAnswer(result.answer);
      setCitations(result.citations || []);
      setQuestion("");
    } catch (err) {
      Alert.alert("Question failed", err.message);
    }
  }

  return (
    <View>
      <Title title="Q&A Chat Interface" subtitle="Ask questions about your university documents" />
      <View style={styles.panel}>
        <Button label="New Chat" variant="dark" onPress={() => { setAnswer(""); setCitations([]); }} />
        <Text style={styles.heading}>Available Documents</Text>
        {docs.length ? docs.map((doc) => <Text key={doc.id} style={styles.docItem}>{doc.title}{"\n"}{doc.department} - {doc.category}</Text>) : <Text>No processed public documents available yet.</Text>}
      </View>
      <View style={styles.panel}>
        {answer ? <Text style={styles.answer}>{answer}</Text> : <Text>Welcome. Ask a question and the assistant will answer using the uploaded documents.</Text>}
        {citations.map((citation) => <Text key={citation.chunkId} style={styles.docItem}>{citation.documentTitle}{"\n"}Chunk ID: {citation.chunkId} | Score: {citation.score}{"\n"}{citation.preview}</Text>)}
      </View>
      <TextInput style={styles.input} value={question} onChangeText={setQuestion} placeholder="Type your question here..." />
      <Button label="Send" variant="dark" onPress={ask} />
    </View>
  );
}

function AdminDashboard({ setScreen }) {
  const [stats, setStats] = useState({ students: 0, documents: 0, questions: 0, answers: 0 });
  useEffect(() => { api.dashboard().then(setStats).catch((err) => Alert.alert("Dashboard", err.message)); }, []);
  return (
    <View>
      <Title title="Admin Dashboard" subtitle="University Portal Management" />
      <View style={styles.metricGrid}>{Object.entries(stats).map(([key, value]) => <View style={styles.metric} key={key}><Text style={styles.heading}>{labelize(key)}</Text><Text style={styles.metricValue}>{value}</Text></View>)}</View>
      <Button label="Upload Documents" variant="dark" onPress={() => setScreen("upload")} />
      <Button label="Document Library" onPress={() => setScreen("documents")} />
    </View>
  );
}

function UploadDocument({ setScreen, setNotice }) {
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({ title: "", department: "", course: "", category: "", description: "", tags: "", isPublic: false });

  async function pickFile() {
    const result = await DocumentPicker.getDocumentAsync({ type: ["application/pdf", "text/plain"] });
    if (!result.canceled) setFile(result.assets[0]);
  }

  async function submit() {
    if (!file) {
      Alert.alert("Upload", "Please choose a file.");
      return;
    }

    const data = new FormData();
    data.append("file", { uri: file.uri, name: file.name, type: file.mimeType || "application/octet-stream" });
    Object.entries(form).forEach(([key, value]) => data.append(key, String(value)));

    try {
      const result = await api.uploadDocument(data);
      setNotice(result.message);
      setScreen("documents");
    } catch (err) {
      Alert.alert("Upload failed", err.message);
    }
  }

  return (
    <Card title="Document Upload" subtitle="Upload course materials and resources">
      <Button label={file ? file.name : "Select File"} onPress={pickFile} />
      {["title", "department", "course", "category", "description", "tags"].map((key) => (
        <Input key={key} label={labelize(key)} multiline={key === "description"} value={form[key]} onChangeText={(value) => setForm({ ...form, [key]: value })} />
      ))}
      <View style={styles.switchRow}><Switch value={form.isPublic} onValueChange={(isPublic) => setForm({ ...form, isPublic })} /><Text>Make this document publicly available</Text></View>
      <Button label="Upload Document" variant="dark" onPress={submit} />
    </Card>
  );
}

function Documents({ setScreen }) {
  const [docs, setDocs] = useState([]);
  useEffect(() => { api.documents().then(setDocs).catch((err) => Alert.alert("Documents", err.message)); }, []);
  return (
    <View>
      <Title title="Document Library" subtitle="Browse and manage course documents" />
      {docs.map((doc) => (
        <View style={styles.panel} key={doc.id}>
          <Text style={styles.heading}>{doc.title}</Text>
          <Text>{doc.department} | {doc.category}</Text>
          <Text>{doc.status}</Text>
          <Button label="Details and Chunks" onPress={() => setScreen(`document:${doc.id}`)} />
        </View>
      ))}
      <Button label="Back to Dashboard" onPress={() => setScreen("dashboard")} />
    </View>
  );
}

function DocumentDetails({ id, setScreen }) {
  const [doc, setDoc] = useState(null);

  useEffect(() => {
    api.documentDetails(id).then(setDoc).catch((err) => Alert.alert("Document", err.message));
  }, [id]);

  if (!doc) {
    return <Text>Loading document...</Text>;
  }

  return (
    <View>
      <Title title="Document Details" subtitle="View document metadata and chunks" />
      <View style={styles.panel}>
        {["title", "department", "course", "category", "status", "fileType", "fileSize", "processingMethod"].map((key) => (
          <Text key={key} style={styles.detailLine}>{labelize(key)}: {doc[key]}</Text>
        ))}
      </View>
      <View style={styles.panel}>
        <Text style={styles.heading}>Chunks</Text>
        {doc.chunks?.length ? doc.chunks.map((chunk) => (
          <Text key={chunk.id} style={styles.docItem}>Chunk #{chunk.chunkIndex}{"\n"}{chunk.text}</Text>
        )) : <Text>No chunks found for this document.</Text>}
      </View>
      <Button label="Back to List" onPress={() => setScreen("documents")} />
    </View>
  );
}

function Card({ title, subtitle, children }) {
  return <View style={styles.card}><Text style={styles.cardTitle}>{title}</Text><Text style={styles.subtitle}>{subtitle}</Text>{children}</View>;
}

function Title({ title, subtitle }) {
  return <View style={styles.titleCard}><Text style={styles.pageTitle}>{title}</Text><Text style={styles.subtitle}>{subtitle}</Text></View>;
}

function Input({ label, ...props }) {
  return <View style={styles.field}><Text style={styles.label}>{label}</Text><TextInput style={[styles.input, props.multiline && styles.textarea]} placeholder={label} {...props} /></View>;
}

function Button({ label, variant, onPress }) {
  return <TouchableOpacity style={[styles.button, variant === "dark" && styles.buttonDark]} onPress={onPress}><Text style={[styles.buttonText, variant === "dark" && styles.buttonTextDark]}>{label}</Text></TouchableOpacity>;
}

function labelize(value) {
  return value.replace(/([A-Z])/g, " $1").replace(/^./, (letter) => letter.toUpperCase());
}

const styles = StyleSheet.create({
  app: { flex: 1, backgroundColor: "#cfd6e1" },
  header: { padding: 18, paddingTop: 54, backgroundColor: "#eef1f5", borderBottomWidth: 2, borderColor: "#8f99aa", gap: 14 },
  brand: { fontSize: 28, fontWeight: "700", color: "#10233f" },
  subtitle: { fontSize: 16, color: "#66758a", marginBottom: 12 },
  shell: { padding: 16, paddingBottom: 40 },
  card: { backgroundColor: "#f7f8fa", borderWidth: 2, borderColor: "#9099aa", padding: 22, gap: 10 },
  cardTitle: { fontSize: 30, fontWeight: "700", color: "#10233f" },
  titleCard: { backgroundColor: "#f7f8fa", borderWidth: 2, borderColor: "#9099aa", padding: 20, marginBottom: 16 },
  pageTitle: { fontSize: 28, fontWeight: "700", color: "#10233f" },
  field: { marginBottom: 12 },
  label: { fontWeight: "700", color: "#10233f", marginBottom: 8 },
  input: { backgroundColor: "#f4f6f9", borderWidth: 1, borderColor: "#9ca6b5", padding: 14, fontSize: 16 },
  textarea: { minHeight: 110, textAlignVertical: "top" },
  button: { backgroundColor: "#ffffff", borderWidth: 1, borderColor: "#9ca6b5", padding: 15, alignItems: "center", marginTop: 8 },
  buttonDark: { backgroundColor: "#142744", borderColor: "#142744" },
  buttonText: { color: "#10233f", fontWeight: "700" },
  buttonTextDark: { color: "#ffffff" },
  success: { backgroundColor: "#e3f2e6", borderWidth: 1, borderColor: "#a5c9ad", color: "#245b2e", padding: 14, marginBottom: 16 },
  switchRow: { flexDirection: "row", alignItems: "center", gap: 10, marginVertical: 12 },
  panel: { backgroundColor: "#ffffff", borderWidth: 1, borderColor: "#a2adba", padding: 16, marginBottom: 16 },
  heading: { fontSize: 18, fontWeight: "700", color: "#10233f", marginBottom: 8 },
  docItem: { borderBottomWidth: 1, borderColor: "#b8c0cc", paddingVertical: 10 },
  detailLine: { marginBottom: 8, color: "#10233f" },
  answer: { fontSize: 16, lineHeight: 24 },
  metricGrid: { gap: 12, marginBottom: 16 },
  metric: { backgroundColor: "#f7f8fa", borderWidth: 2, borderColor: "#9099aa", padding: 18 },
  metricValue: { fontSize: 34, fontWeight: "700", color: "#10233f" },
});
