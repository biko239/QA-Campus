using System.Text;
using System.Text.Json;

namespace Fyp.Services
{
    public class MlEmbeddingService
    {
        private readonly IHttpClientFactory _http;
        private readonly IConfiguration _cfg;

        public MlEmbeddingService(IHttpClientFactory http, IConfiguration cfg)
        {
            _http = http;
            _cfg = cfg;
        }

        public async Task<float[]> EmbedAsync(string text, string mode = "document")
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));

            var client = _http.CreateClient();

            var baseUrl = (_cfg["AiService:BaseUrl"] ?? _cfg["MlService:BaseUrl"] ?? "http://127.0.0.1:8000")
                .TrimEnd('/');

            var body = JsonSerializer.Serialize(new { text, mode });

            var response = await client.PostAsync(
                $"{baseUrl}/embed",
                new StringContent(body, Encoding.UTF8, "application/json")
            );

            var content = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"AI embedding service error: {(int)response.StatusCode} - {content}");

            using var doc = JsonDocument.Parse(content);

            if (!doc.RootElement.TryGetProperty("embedding", out var embeddingElement))
                throw new Exception("AI service response does not contain 'embedding'.");

            var list = new List<float>();

            foreach (var item in embeddingElement.EnumerateArray())
            {
                list.Add(item.GetSingle());
            }

            return list.ToArray();
        }
    }
}
