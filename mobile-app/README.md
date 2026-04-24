# Mobile App

This Expo app uses the same ASP.NET Core backend API as the React web client.

## Run

```powershell
npm install
npm run start
```

The default API URL is `http://localhost:5110`. On a physical phone, `localhost` points to the phone, not your PC. Use your computer's LAN IP address instead:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://YOUR-PC-IP:5110"
npm run start
```
