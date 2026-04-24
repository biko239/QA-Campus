# QA Campus

QA Campus is an ASP.NET Core, React, and Expo project for university document upload, processing, and question answering.

## Project Structure

- `Controllers/` - backend API endpoints for auth, admin, and student features.
- `Data/`, `Models/`, `Migrations/` - Entity Framework Core database layer.
- `Services/` - document processing, authentication, analytics, RAG, Qdrant, and startup orchestration.
- `client-web/` - React web client served by the ASP.NET Core backend after build.
- `mobile-app/` - Expo mobile client.
- `ai-training/` - scripts for exporting data and training retriever/generator adapter models.
- `wwwroot/` - static ASP.NET Core web root placeholder.

## Run From Visual Studio

1. Open `Fyp.sln`.
2. Select the `QA Campus` launch profile.
3. Press the green Start button.

Visual Studio will build the React client, start the ASP.NET Core backend, and open the app at `http://localhost:5110`.

## Requirements

- .NET 8 SDK
- Node.js and npm
- MySQL running with the connection configured in `appsettings.json`
- Optional for full RAG: local AI service folder and Docker/Qdrant

Generated folders such as `bin/`, `obj/`, `node_modules/`, `client-web/dist/`, `.vs/`, uploads, logs, and runtime keys are intentionally ignored and should be recreated locally when needed.
