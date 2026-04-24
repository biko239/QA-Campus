using System.Diagnostics;

namespace Fyp.Services
{
    public class StartupOrchestratorService : IHostedService
    {
        private readonly IConfiguration _cfg;
        private Process? _aiProcess;

        public StartupOrchestratorService(IConfiguration cfg)
        {
            _cfg = cfg;
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("Starting local AI services...");

            try
            {
                await StartQdrantAsync(cancellationToken);
                await StartAIAsync(cancellationToken);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error starting local services: " + ex.Message);
            }
        }

        private async Task StartQdrantAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("Checking Qdrant...");

            var qdrantUrl = (_cfg["Qdrant:Url"] ?? "http://localhost:6333").TrimEnd('/');
            if (await IsHttpAvailableAsync($"{qdrantUrl}/collections", cancellationToken))
            {
                Console.WriteLine("Qdrant is already running.");
                return;
            }

            var container = _cfg["Startup:QdrantContainer"] ?? "qdrant-fyp";

            try
            {
                var started = await RunProcessAsync("docker", new[] { "start", container }, cancellationToken);
                var created = false;
                if (!started)
                {
                    Console.WriteLine($"Qdrant container '{container}' was not found. Trying to create it.");
                    created = await RunProcessAsync("docker", new[] { "run", "-d", "--name", container, "-p", "6333:6333", "qdrant/qdrant" }, cancellationToken);
                }

                if (!started && !created)
                {
                    Console.WriteLine("Qdrant did not start. Chat will use database text fallback.");
                    return;
                }

                if (await WaitForHttpAsync($"{qdrantUrl}/collections", TimeSpan.FromSeconds(10), cancellationToken))
                    Console.WriteLine("Qdrant is ready.");
                else
                    Console.WriteLine("Qdrant did not become reachable. Chat will use database text fallback.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Qdrant could not be started. Chat will use database text fallback. " + ex.Message);
            }
        }

        private async Task StartAIAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("Checking AI service...");

            var aiBaseUrl = (_cfg["AiService:BaseUrl"] ?? _cfg["MlService:BaseUrl"] ?? "http://127.0.0.1:8000").TrimEnd('/');
            if (await IsHttpAvailableAsync($"{aiBaseUrl}/docs", cancellationToken))
            {
                Console.WriteLine("AI service is already running.");
                return;
            }

            var aiPath = _cfg["Startup:AiFolder"] ?? @"C:\Users\alian.ALI\Desktop\ai";
            var pythonExe = _cfg["Startup:PythonExe"] ?? "python";

            Console.WriteLine("AI Path: " + aiPath);

            if (!Directory.Exists(aiPath))
            {
                Console.WriteLine("AI folder not found.");
                return;
            }

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                WorkingDirectory = aiPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = false
            };

            psi.ArgumentList.Add("-m");
            psi.ArgumentList.Add("uvicorn");
            psi.ArgumentList.Add("app:app");
            psi.ArgumentList.Add("--host");
            psi.ArgumentList.Add("127.0.0.1");
            psi.ArgumentList.Add("--port");
            psi.ArgumentList.Add("8000");

            _aiProcess = new Process { StartInfo = psi };
            _aiProcess.OutputDataReceived += (s, e) => { if (e.Data != null) Console.WriteLine("[AI] " + e.Data); };
            _aiProcess.ErrorDataReceived += (s, e) => { if (e.Data != null) Console.WriteLine("[AI] " + e.Data); };

            _aiProcess.Start();
            _aiProcess.BeginOutputReadLine();
            _aiProcess.BeginErrorReadLine();
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("Stopping local AI services...");

            try
            {
                if (_aiProcess != null && !_aiProcess.HasExited)
                {
                    _aiProcess.Kill();
                    Console.WriteLine("AI service stopped.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error stopping services: " + ex.Message);
            }

            return Task.CompletedTask;
        }

        private static async Task<bool> RunProcessAsync(string fileName, string[] args, CancellationToken cancellationToken)
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };

            foreach (var arg in args)
                psi.ArgumentList.Add(arg);

            using var process = Process.Start(psi);
            if (process == null)
                return false;

            await process.WaitForExitAsync(cancellationToken);

            var output = await process.StandardOutput.ReadToEndAsync(cancellationToken);
            var error = await process.StandardError.ReadToEndAsync(cancellationToken);

            if (!string.IsNullOrWhiteSpace(output))
                Console.WriteLine(output.Trim());

            if (!string.IsNullOrWhiteSpace(error))
                Console.WriteLine(error.Trim());

            return process.ExitCode == 0;
        }

        private static async Task<bool> WaitForHttpAsync(string url, TimeSpan timeout, CancellationToken cancellationToken)
        {
            var deadline = DateTime.UtcNow.Add(timeout);
            while (DateTime.UtcNow < deadline && !cancellationToken.IsCancellationRequested)
            {
                if (await IsHttpAvailableAsync(url, cancellationToken))
                    return true;

                await Task.Delay(1000, cancellationToken);
            }

            return false;
        }

        private static async Task<bool> IsHttpAvailableAsync(string url, CancellationToken cancellationToken)
        {
            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
                using var response = await client.GetAsync(url, cancellationToken);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }
    }
}
