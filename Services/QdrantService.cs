using System.Text;
using System.Text.Json;

namespace Fyp.Services
{
    public class QdrantSearchResult
    {
        public long Id { get; set; }
        public float Score { get; set; }
    }

    public class QdrantService
    {
        private readonly IHttpClientFactory _http;
        private readonly IConfiguration _cfg;

        public QdrantService(IHttpClientFactory http, IConfiguration cfg)
        {
            _http = http;
            _cfg = cfg;
        }

        private string BaseUrl => _cfg["Qdrant:Url"] ?? "http://localhost:6333";
        private string Collection => _cfg["Qdrant:Collection"] ?? "fyp_chunks";
        private int VectorSize => int.Parse(_cfg["Qdrant:VectorSize"] ?? "384");

        private HttpClient CreateClient()
        {
            var client = _http.CreateClient();
            client.BaseAddress = new Uri(BaseUrl);
            client.Timeout = TimeSpan.FromSeconds(10);
            return client;
        }

        public async Task EnsureCollectionAsync()
        {
            var client = CreateClient();

            var getResponse = await client.GetAsync($"/collections/{Collection}");
            if (getResponse.IsSuccessStatusCode)
                return;

            var body = new
            {
                vectors = new
                {
                    size = VectorSize,
                    distance = "Cosine"
                }
            };

            var response = await client.PutAsync(
                $"/collections/{Collection}",
                new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json")
            );

            var content = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Qdrant create collection error: {(int)response.StatusCode} - {content}");
        }

        public async Task UpsertAsync(long pointId, float[] vector, object payload)
        {
            var client = CreateClient();

            var body = new
            {
                points = new object[]
                {
                    new
                    {
                        id = pointId,
                        vector = vector,
                        payload = payload
                    }
                }
            };

            var response = await client.PutAsync(
                $"/collections/{Collection}/points?wait=true",
                new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json")
            );

            var content = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Qdrant upsert error: {(int)response.StatusCode} - {content}");
        }

        public async Task<List<QdrantSearchResult>> SearchAsync(float[] vector, int topK)
        {
            var client = CreateClient();

            var body = new
            {
                vector = vector,
                limit = topK,
                with_payload = false,
                with_vector = false
            };

            var response = await client.PostAsync(
                $"/collections/{Collection}/points/search",
                new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json")
            );

            var content = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
                throw new Exception($"Qdrant search error: {(int)response.StatusCode} - {content}");

            using var doc = JsonDocument.Parse(content);
            var results = new List<QdrantSearchResult>();

            if (doc.RootElement.TryGetProperty("result", out var resultArray))
            {
                foreach (var item in resultArray.EnumerateArray())
                {
                    results.Add(new QdrantSearchResult
                    {
                        Id = item.GetProperty("id").GetInt64(),
                        Score = item.GetProperty("score").GetSingle()
                    });
                }
            }

            return results;
        }
    }
}
