namespace Fyp.Services
{
    public class ChunkingService
    {
        private readonly IConfiguration _cfg;

        public ChunkingService(IConfiguration cfg)
        {
            _cfg = cfg;
        }

        public List<string> Chunk(string text)
        {
            int wordsPerChunk = int.Parse(_cfg["Rag:WordsPerChunk"] ?? "450");
            int overlapWords = int.Parse(_cfg["Rag:OverlapWords"] ?? "50");

            var words = text.Split(
                new[] { ' ', '\n', '\r', '\t' },
                StringSplitOptions.RemoveEmptyEntries
            );

            var chunks = new List<string>();
            int start = 0;

            while (start < words.Length)
            {
                int end = Math.Min(start + wordsPerChunk, words.Length);
                chunks.Add(string.Join(" ", words[start..end]));

                if (end == words.Length)
                    break;

                start = Math.Max(0, end - overlapWords);
            }

            return chunks;
        }
    }
}