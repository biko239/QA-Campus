# React Web Client

This React client uses the existing ASP.NET Core backend API.

## Run

```powershell
npm install
npm run dev
```

The default API URL is `http://localhost:5110`. To use another backend URL:

```powershell
$env:VITE_API_BASE_URL="http://localhost:5110"
npm run dev
```
