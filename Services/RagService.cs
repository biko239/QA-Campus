using System.Globalization;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Fyp.Data;
using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Services
{
    public class RagService
    {
        private readonly AppDbContext _db;
        private readonly IConfiguration _cfg;
        private readonly DocumentService _docs;
        private readonly TextExtractService _extract;
        private readonly ChunkingService _chunker;
        private readonly MlEmbeddingService _ml;
        private readonly QdrantService _qdrant;
        private readonly HttpClient _http;
        private readonly ILogger<RagService> _logger;

        public RagService(
            AppDbContext db,
            IConfiguration cfg,
            DocumentService docs,
            TextExtractService extract,
            ChunkingService chunker,
            MlEmbeddingService ml,
            QdrantService qdrant,
            IHttpClientFactory httpClientFactory,
            ILogger<RagService> logger)
        {
            _db = db;
            _cfg = cfg;
            _docs = docs;
            _extract = extract;
            _chunker = chunker;
            _ml = ml;
            _qdrant = qdrant;
            _http = httpClientFactory.CreateClient();
            _http.Timeout = TimeSpan.FromSeconds(8);
            _logger = logger;
        }

        public async Task ProcessDocumentAsync(int documentId)
        {
            var doc = await _db.Documents.FirstOrDefaultAsync(d => d.Id == documentId);
            if (doc == null)
                throw new Exception("Document not found.");

            doc.Status = "PROCESSING";
            doc.ProcessingStartedAt = DateTime.UtcNow;
            await _db.SaveChangesAsync();

            try
            {
                var path = Path.Combine(_docs.UploadFolder(), doc.StoredFileName);

                var text = await _extract.ExtractAsync(path);
                if (string.IsNullOrWhiteSpace(text))
                    throw new Exception("No text could be extracted from the document.");

                var chunks = _chunker.Chunk(text);

                var oldChunks = _db.DocumentChunks.Where(c => c.DocumentId == doc.Id);
                _db.DocumentChunks.RemoveRange(oldChunks);
                await _db.SaveChangesAsync();

                var chunkRows = chunks.Select((chunkText, index) => new DocumentChunk
                {
                    DocumentId = doc.Id,
                    ChunkIndex = index,
                    Text = chunkText,
                    CreatedAt = DateTime.UtcNow
                }).ToList();

                _db.DocumentChunks.AddRange(chunkRows);
                await _db.SaveChangesAsync();

                var vectorCount = await TryIndexChunksInQdrantAsync(doc, chunkRows);

                doc.Status = "READY";
                doc.ProcessingMethod = vectorCount > 0 ? "PDF_RAG_PIPELINE" : "TEXT_CHUNK_FALLBACK";
                doc.ProcessingCompletedAt = DateTime.UtcNow;
                await _db.SaveChangesAsync();
            }
            catch
            {
                doc.Status = "FAILED";
                doc.ProcessingCompletedAt = DateTime.UtcNow;
                await _db.SaveChangesAsync();
                throw;
            }
        }

        public async Task<(Answer answer, List<(DocumentChunk chunk, float score)> citations)> AskAsync(int userId, string questionText)
        {
            var question = new Question
            {
                UserId = userId,
                Text = questionText,
                AskedAt = DateTime.UtcNow
            };

            _db.Questions.Add(question);
            await _db.SaveChangesAsync();

            var documents = await GetReadyPublicDocumentsAsync();
            var explicitDocumentMatches = FindExplicitDocumentMatches(documents, questionText);
            var recentDocument = await GetRecentDocumentContextAsync(userId, question.Id, documents);
            var targetDocument = explicitDocumentMatches.Count == 1 ? explicitDocumentMatches[0] : recentDocument;

            if (targetDocument == null && LooksLikeCalendarQuestion(questionText))
            {
                var calendarDocuments = documents.Where(IsCalendarDocument).ToList();
                if (calendarDocuments.Count == 1)
                    targetDocument = calendarDocuments[0];
            }

            if (IsSmallTalk(questionText))
            {
                var smallTalkAnswer = BuildSmallTalkAnswer(questionText);
                var answerSmall = await SaveAnswerAsync(question.Id, smallTalkAnswer, 1.0f);
                return (answerSmall, new List<(DocumentChunk chunk, float score)>());
            }

            var topicClarification = TryBuildTopicClarification(questionText, documents, explicitDocumentMatches, targetDocument);
            if (!string.IsNullOrWhiteSpace(topicClarification))
            {
                var clarificationAnswer = await SaveAnswerAsync(question.Id, topicClarification, 0.95f);
                return (clarificationAnswer, new List<(DocumentChunk chunk, float score)>());
            }

            int topK = int.Parse(_cfg["Rag:TopK"] ?? "4");
            var citations = await TryVectorSearchAsync(questionText, topK, targetDocument?.Id);

            if (citations.Count == 0)
                citations = await SearchDatabaseChunksAsync(questionText, topK, targetDocument?.Id);

            string finalAnswer = await GenerateAnswerAsync(questionText, citations);
            var confidence = citations.Count > 0 ? citations[0].score : 0.1f;
            var answer = await SaveAnswerAsync(question.Id, finalAnswer, confidence);

            foreach (var citation in citations)
            {
                _db.ChunkUsages.Add(new ChunkUsage
                {
                    AnswerId = answer.Id,
                    ChunkId = citation.chunk.Id,
                    Score = citation.score
                });
            }

            await _db.SaveChangesAsync();

            return (answer, citations);
        }

        private async Task<int> TryIndexChunksInQdrantAsync(Document doc, List<DocumentChunk> chunks)
        {
            try
            {
                await _qdrant.EnsureCollectionAsync();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Qdrant is unavailable. Document {DocumentId} will use database text fallback.", doc.Id);
                return 0;
            }

            var indexed = 0;
            foreach (var chunkRow in chunks)
            {
                try
                {
                    var embedding = await _ml.EmbedAsync(chunkRow.Text);

                    await _qdrant.UpsertAsync(
                        chunkRow.Id,
                        embedding,
                        new
                        {
                            documentId = doc.Id,
                            chunkId = chunkRow.Id,
                            chunkIndex = chunkRow.ChunkIndex,
                            title = doc.Title,
                            department = doc.Department,
                            category = doc.Category,
                            isPublic = doc.IsPublic,
                            status = doc.Status
                        }
                    );

                    indexed++;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Could not vector-index chunk {ChunkId}. Database fallback will still work.", chunkRow.Id);
                }
            }

            return indexed;
        }

        private async Task<List<(DocumentChunk chunk, float score)>> TryVectorSearchAsync(string questionText, int topK, int? targetDocumentId)
        {
            try
            {
                var questionEmbedding = await _ml.EmbedAsync(questionText);
                await _qdrant.EnsureCollectionAsync();
                var matches = await _qdrant.SearchAsync(questionEmbedding, topK);
                return await HydrateVectorMatchesAsync(matches, targetDocumentId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Vector search failed. Falling back to database text search.");
                return new List<(DocumentChunk chunk, float score)>();
            }
        }

        private async Task<List<(DocumentChunk chunk, float score)>> HydrateVectorMatchesAsync(List<QdrantSearchResult> matches, int? targetDocumentId)
        {
            var chunkIds = matches
                .Select(x => (int)x.Id)
                .ToList();

            if (chunkIds.Count == 0)
                return new List<(DocumentChunk chunk, float score)>();

            var chunks = await _db.DocumentChunks
                .Include(c => c.Document)
                .Where(c =>
                    chunkIds.Contains(c.Id) &&
                    c.Document != null &&
                    c.Document.IsPublic &&
                    c.Document.Status == "READY" &&
                    (!targetDocumentId.HasValue || c.DocumentId == targetDocumentId.Value))
                .ToListAsync();

            var citations = new List<(DocumentChunk chunk, float score)>();

            foreach (var match in matches)
            {
                var row = chunks.FirstOrDefault(c => c.Id == (int)match.Id);
                if (row != null)
                    citations.Add((row, match.Score));
            }

            return citations;
        }

        private async Task<List<(DocumentChunk chunk, float score)>> SearchDatabaseChunksAsync(string questionText, int topK, int? targetDocumentId)
        {
            var terms = ExpandQueryTerms(Tokenize(questionText))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (terms.Count == 0)
                return new List<(DocumentChunk chunk, float score)>();

            var candidates = await _db.DocumentChunks
                .Include(c => c.Document)
                .Where(c =>
                    c.Document != null &&
                    c.Document.IsPublic &&
                    c.Document.Status == "READY" &&
                    (!targetDocumentId.HasValue || c.DocumentId == targetDocumentId.Value))
                .ToListAsync();

            var scored = candidates
                .Select(chunk => (chunk, score: ScoreChunk(chunk, terms, questionText)))
                .Where(item => item.score > 0)
                .OrderByDescending(item => item.score)
                .ThenBy(item => item.chunk.ChunkIndex)
                .Take(topK)
                .ToList();

            if (scored.Count > 0)
                return scored;

            if (targetDocumentId.HasValue && IsSummaryLike(questionText))
            {
                return candidates
                    .OrderBy(chunk => chunk.ChunkIndex)
                    .Take(topK)
                    .Select(chunk => (chunk, score: 0.5f))
                    .ToList();
            }

            return new List<(DocumentChunk chunk, float score)>();
        }

        private async Task<Answer> SaveAnswerAsync(int questionId, string text, float confidence)
        {
            var answer = new Answer
            {
                QuestionId = questionId,
                Text = text,
                Confidence = confidence,
                CreatedAt = DateTime.UtcNow
            };

            _db.Answers.Add(answer);
            await _db.SaveChangesAsync();
            return answer;
        }

        private async Task<List<Document>> GetReadyPublicDocumentsAsync()
        {
            return await _db.Documents
                .Include(d => d.Chunks)
                .Where(d => d.IsPublic && d.Status == "READY")
                .ToListAsync();
        }

        private static List<Document> FindExplicitDocumentMatches(List<Document> documents, string questionText)
        {
            var normalizedQuestion = NormalizeForMatch(questionText);
            var questionTokens = Tokenize(questionText)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            return documents
                .Select(doc => new
                {
                    Document = doc,
                    Score = ScoreExplicitDocumentMatch(doc, normalizedQuestion, questionTokens)
                })
                .Where(item => item.Score > 0)
                .OrderByDescending(item => item.Score)
                .Select(item => item.Document)
                .ToList();
        }

        private static int ScoreExplicitDocumentMatch(Document doc, string normalizedQuestion, List<string> questionTokens)
        {
            var score = 0;
            var title = NormalizeForMatch(doc.Title);
            var originalFileName = NormalizeForMatch(Path.GetFileNameWithoutExtension(doc.OriginalFileName));
            var category = NormalizeForMatch(doc.Category);

            if (!string.IsNullOrWhiteSpace(title))
            {
                if (normalizedQuestion == title)
                    score += 100;
                else if (normalizedQuestion.Contains(title) || title.Contains(normalizedQuestion))
                    score += 60;
            }

            if (!string.IsNullOrWhiteSpace(originalFileName))
            {
                if (normalizedQuestion.Contains(originalFileName) || originalFileName.Contains(normalizedQuestion))
                    score += 30;
            }

            foreach (var token in questionTokens)
            {
                var normalizedToken = NormalizeForMatch(token);
                if (normalizedToken.Length < 3)
                    continue;

                if (title.Split(' ', StringSplitOptions.RemoveEmptyEntries).Contains(normalizedToken))
                    score += 20;

                if (originalFileName.Contains(normalizedToken))
                    score += 10;

                if (category.Contains(normalizedToken))
                    score += 4;
            }

            return score;
        }

        private async Task<Document?> GetRecentDocumentContextAsync(int userId, int currentQuestionId, List<Document> documents)
        {
            var recentQuestions = await _db.Questions
                .Include(q => q.Answer)
                    .ThenInclude(a => a!.ChunkUsages)
                        .ThenInclude(cu => cu.Chunk)
                            .ThenInclude(c => c!.Document)
                .Where(q => q.UserId == userId && q.Id != currentQuestionId)
                .OrderByDescending(q => q.Id)
                .Take(8)
                .ToListAsync();

            foreach (var question in recentQuestions)
            {
                var citedDocument = question.Answer?.ChunkUsages
                    .Select(cu => cu.Chunk?.Document)
                    .FirstOrDefault(doc => doc != null && doc.IsPublic && doc.Status == "READY");

                if (citedDocument != null)
                    return documents.FirstOrDefault(doc => doc.Id == citedDocument.Id);

                var text = $"{question.Text} {question.Answer?.Text}";
                var explicitMatches = FindExplicitDocumentMatches(documents, text);
                if (explicitMatches.Count == 1)
                    return explicitMatches[0];
            }

            return null;
        }

        private static string? TryBuildTopicClarification(
            string questionText,
            List<Document> documents,
            List<Document> explicitDocumentMatches,
            Document? recentDocument)
        {
            if (!IsBroadTopicQuery(questionText))
                return null;

            if (explicitDocumentMatches.Count == 1 && IsDocumentNameQuery(explicitDocumentMatches[0], questionText))
                return BuildDocumentSuggestionAnswer(explicitDocumentMatches[0], questionText);

            if (explicitDocumentMatches.Count > 1)
                return BuildDocumentDisambiguationAnswer(questionText, explicitDocumentMatches);

            if (recentDocument != null)
                return null;

            var terms = ExpandQueryTerms(Tokenize(questionText))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
            if (terms.Count == 0)
                return null;

            var matches = documents
                .Select(doc => new
                {
                    Document = doc,
                    Score = ScoreDocumentTopicMatch(doc, terms, questionText)
                })
                .Where(item => item.Score > 0)
                .OrderByDescending(item => item.Score)
                .ThenBy(item => item.Document.Title)
                .Take(4)
                .ToList();

            if (matches.Count == 0)
                return null;

            if (matches.Count > 1)
                return BuildDocumentDisambiguationAnswer(questionText, matches.Select(m => m.Document).ToList());

            return BuildDocumentSuggestionAnswer(matches[0].Document, questionText);
        }

        private static bool IsDocumentNameQuery(Document doc, string questionText)
        {
            var normalizedQuestion = NormalizeForMatch(questionText);
            var title = NormalizeForMatch(doc.Title);
            var fileName = NormalizeForMatch(Path.GetFileNameWithoutExtension(doc.OriginalFileName));

            return normalizedQuestion == title ||
                   normalizedQuestion == fileName ||
                   (!string.IsNullOrWhiteSpace(title) && title.Contains(normalizedQuestion) && Tokenize(questionText).Count <= 2) ||
                   (!string.IsNullOrWhiteSpace(fileName) && fileName.Contains(normalizedQuestion) && Tokenize(questionText).Count <= 2);
        }

        private static string BuildDocumentDisambiguationAnswer(string questionText, List<Document> documents)
        {
            var options = documents
                .Take(5)
                .Select((doc, index) => $"{index + 1}. {doc.Title} ({doc.Category})")
                .ToList();

            return $"I found more than one uploaded document related to \"{questionText.Trim()}\".\n\nWhich one do you mean?\n\n{string.Join("\n", options)}\n\nYou can ask, for example: \"summarize {documents[0].Title}\" or \"what are the important dates in {documents[0].Title}?\"";
        }

        private static bool IsBroadTopicQuery(string text)
        {
            var normalized = (text ?? "").Trim();
            if (normalized.Length == 0 || normalized.Contains('?'))
                return false;

            var lower = normalized.ToLowerInvariant();
            var directQuestionStarters = new[]
            {
                "what ", "how ", "why ", "when ", "where ", "who ", "which ",
                "explain ", "summarize ", "summary ", "list ", "give ", "show ",
                "compare ", "define ", "describe "
            };

            if (directQuestionStarters.Any(lower.StartsWith))
                return false;

            var tokens = Tokenize(normalized)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            return tokens.Count is > 0 and <= 3;
        }

        private static bool IsSummaryLike(string text)
        {
            var normalized = NormalizeForMatch(text);
            return normalized.Contains("summary") ||
                   normalized.Contains("summarize") ||
                   normalized.Contains("summarise") ||
                   normalized.Contains("overview");
        }

        private static float ScoreDocumentTopicMatch(Document doc, List<string> terms, string originalText)
        {
            var title = NormalizeForMatch(doc.Title);
            var metadata = NormalizeForMatch($"{doc.Department} {doc.Course} {doc.Category} {doc.Tags}");
            var query = NormalizeForMatch(originalText);

            float score = 0;

            if (title == query)
                score += 10;
            else if (title.Contains(query) || query.Contains(title))
                score += 6;

            foreach (var term in terms)
            {
                var normalizedTerm = NormalizeForMatch(term);

                if (title.Contains(normalizedTerm))
                    score += 5;

                if (metadata.Contains(normalizedTerm))
                    score += 2;

                if (doc.Chunks.Any(chunk => NormalizeForMatch(chunk.Text).Contains(normalizedTerm)))
                    score += 1;
            }

            return score;
        }

        private static string BuildDocumentSuggestionAnswer(Document doc, string questionText)
        {
            var title = string.IsNullOrWhiteSpace(doc.Title) ? questionText.Trim() : doc.Title.Trim();
            var topics = ExtractSuggestionTopics(doc);
            var suggestions = topics
                .Take(5)
                .Select((topic, index) => $"{index + 1}. {BuildSuggestedQuestion(title, topic)}")
                .ToList();

            if (suggestions.Count == 0)
            {
                suggestions.Add($"1. Can you summarize {title}?");
                suggestions.Add($"2. What are the main rules in {title}?");
                suggestions.Add($"3. What responsibilities or obligations does {title} describe?");
                suggestions.Add($"4. What procedures should I know about in {title}?");
            }

            return $"I found the uploaded document \"{title}\". What exactly would you like to know about it?\n\nTry asking:\n\n{string.Join("\n", suggestions)}";
        }

        private static List<string> ExtractSuggestionTopics(Document doc)
        {
            if (IsCalendarDocument(doc))
            {
                return new List<string>
                {
                    "exam dates",
                    "semester start dates",
                    "holidays and no-class days",
                    "diploma ceremonies",
                    "summer trimester",
                    "summary"
                };
            }

            var text = string.Join(" ", doc.Chunks
                .OrderBy(c => c.ChunkIndex)
                .Take(8)
                .Select(c => c.Text ?? ""));

            var lower = text.ToLowerInvariant();
            var topics = new List<string>();

            AddTopicIfPresent(topics, lower, "objective", "objective");
            AddTopicIfPresent(topics, lower, "aim", "aim");
            AddTopicIfPresent(topics, lower, "scope", "scope");
            AddTopicIfPresent(topics, lower, "responsibilit", "responsibilities");
            AddTopicIfPresent(topics, lower, "rules and regulations", "rules and regulations");
            AddTopicIfPresent(topics, lower, "implementation", "implementation");
            AddTopicIfPresent(topics, lower, "access request", "access requests");
            AddTopicIfPresent(topics, lower, "recording", "recordings");
            AddTopicIfPresent(topics, lower, "retention", "retention period");
            AddTopicIfPresent(topics, lower, "obligation", "obligations");
            AddTopicIfPresent(topics, lower, "election", "elections");
            AddTopicIfPresent(topics, lower, "committee", "committee");
            AddTopicIfPresent(topics, lower, "membership", "membership");
            AddTopicIfPresent(topics, lower, "password", "password requirements");
            AddTopicIfPresent(topics, lower, "software", "software installation");
            AddTopicIfPresent(topics, lower, "data protection", "data protection");
            AddTopicIfPresent(topics, lower, "confidential", "confidentiality");

            if (!string.IsNullOrWhiteSpace(doc.Category))
                topics.Add(doc.Category.Trim());

            topics.Add("summary");
            topics.Add("main rules");
            topics.Add("procedures");
            topics.Add("responsibilities");

            return topics
                .Where(topic => !string.IsNullOrWhiteSpace(topic))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        private static void AddTopicIfPresent(List<string> topics, string text, string needle, string topic)
        {
            if (text.Contains(needle, StringComparison.OrdinalIgnoreCase))
                topics.Add(topic);
        }

        private static string BuildSuggestedQuestion(string title, string topic)
        {
            return topic.ToLowerInvariant() switch
            {
                "exam dates" => $"When are the exams in {title}?",
                "semester start dates" => $"When do the semesters start in {title}?",
                "holidays and no-class days" => $"What holidays or no-class days are listed in {title}?",
                "diploma ceremonies" => $"When are the diploma ceremonies in {title}?",
                "summer trimester" => $"When does the summer trimester start in {title}?",
                "objective" => $"What is the objective of {title}?",
                "aim" => $"What is the aim of {title}?",
                "scope" => $"What is the scope of {title}?",
                "responsibilities" => $"Who is responsible for {title}?",
                "retention period" => $"What is the retention period in {title}?",
                "access requests" => $"How are access requests handled in {title}?",
                "summary" => $"Can you summarize {title}?",
                "main rules" => $"What are the main rules in {title}?",
                "procedures" => $"What procedures does {title} describe?",
                _ => $"What does {title} say about {topic}?"
            };
        }

        private static bool IsCalendarDocument(Document doc)
        {
            var metadata = $"{doc.Title} {doc.OriginalFileName} {doc.Category} {doc.Tags}".ToLowerInvariant();
            return metadata.Contains("calendar") || metadata.Contains("calendrier");
        }

        private static bool LooksLikeCalendarQuestion(string text)
        {
            var normalized = NormalizeForMatch(text);
            var tokens = Tokenize(text)
                .Select(NormalizeForMatch)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            var hasMonth = ExtractMonthNumbers(normalized).Count > 0;
            var hasCalendarTerm = tokens.Overlaps(new[]
            {
                "calendar", "calendrier", "deadline", "deadlines", "date", "dates",
                "semester", "semesters", "semestre", "semestres", "holiday", "holidays",
                "vacation", "break", "rdd", "diploma", "diplomas", "graduation",
                "summer", "trimester", "trim"
            });
            var hasExamTerm = tokens.Overlaps(new[] { "exam", "exams", "examen", "examens", "examination", "examinations" });
            var asksWhen = normalized.Contains("when") || normalized.Contains("date") || normalized.Contains("dates");

            return hasMonth || hasCalendarTerm || (hasExamTerm && asksWhen);
        }

        private bool IsSmallTalk(string text)
        {
            var t = (text ?? "").Trim().ToLower();
            return t == "hello" || t == "hi" || t == "hey" ||
                   t == "thanks" || t == "thank you" ||
                   t == "bye" || t == "goodbye";
        }

        private static string BuildSmallTalkAnswer(string text)
        {
            var t = (text ?? "").Trim().ToLowerInvariant();

            if (t is "thanks" or "thank you")
                return "You're welcome. Ask me anything about the uploaded university documents.";

            if (t is "bye" or "goodbye")
                return "Goodbye. You can come back anytime with a university document question.";

            return "Hello. Ask me a question about the uploaded university documents and I will answer using the available sources.";
        }

        private async Task<string> GenerateAnswerAsync(
            string questionText,
            List<(DocumentChunk chunk, float score)> citations)
        {
            if (citations.Count == 0)
                return "I could not find enough reliable information in the uploaded university documents.";

            var calendarCitations = citations
                .Where(c => c.chunk.Document != null && IsCalendarDocument(c.chunk.Document))
                .ToList();
            if (calendarCitations.Count > 0 && LooksLikeCalendarQuestion(questionText))
                return BuildCalendarExtractiveAnswer(calendarCitations, questionText);

            if (citations.All(c => c.chunk.Document != null && IsCalendarDocument(c.chunk.Document)))
                return BuildCalendarExtractiveAnswer(citations, questionText);

            var aiBaseUrl = (_cfg["AiService:BaseUrl"] ?? _cfg["MlService:BaseUrl"] ?? "http://127.0.0.1:8000").TrimEnd('/');

            var request = new GenerateRequest
            {
                Question = questionText,
                Chunks = citations
                    .Take(4)
                    .Select(c => new GenerateChunkItem
                    {
                        ChunkId = c.chunk.Id,
                        DocumentTitle = c.chunk.Document?.Title ?? "Unknown Document",
                        Text = Preview(c.chunk.Text, 1200),
                        Score = c.score
                    })
                    .ToList()
            };

            try
            {
                var response = await _http.PostAsJsonAsync($"{aiBaseUrl}/generate", request);

                if (!response.IsSuccessStatusCode)
                {
                    var raw = await response.Content.ReadAsStringAsync();
                    _logger.LogWarning("AI generate service returned {StatusCode}: {Body}", response.StatusCode, raw);
                    return BuildExtractiveAnswer(citations, questionText);
                }

                var result = await response.Content.ReadFromJsonAsync<GenerateResponse>();

                if (result == null || string.IsNullOrWhiteSpace(result.Answer) || !result.Supported)
                    return BuildExtractiveAnswer(citations, questionText);

                return result.Answer.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "AI generate service failed. Returning extractive answer.");
                return BuildExtractiveAnswer(citations, questionText);
            }
        }

        private static string BuildExtractiveAnswer(List<(DocumentChunk chunk, float score)> citations, string questionText)
        {
            if (citations.Count > 0 && citations.All(c => c.chunk.Document != null && IsCalendarDocument(c.chunk.Document)))
                return BuildCalendarExtractiveAnswer(citations, questionText);

            if (IsSummaryLike(questionText))
                return BuildSummaryAnswer(citations);

            var terms = ExpandQueryTerms(Tokenize(questionText))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var parts = citations
                .Take(2)
                .Select((citation, index) =>
                {
                    var title = citation.chunk.Document?.Title ?? "uploaded document";
                    var snippet = ExtractBestSnippet(citation.chunk.Text, terms, 700);
                    return $"{index + 1}. From {title}: {snippet}";
                });

            return "I found relevant information in the uploaded documents:\n\n" + string.Join("\n\n", parts);
        }

        private static string BuildSummaryAnswer(List<(DocumentChunk chunk, float score)> citations)
        {
            var documentGroups = citations
                .Where(citation => citation.chunk.Document != null)
                .GroupBy(citation => citation.chunk.Document!.Id)
                .Take(2)
                .ToList();

            var parts = new List<string>();

            foreach (var group in documentGroups)
            {
                var document = group.First().chunk.Document!;
                var text = string.Join(" ", group
                    .OrderBy(citation => citation.chunk.ChunkIndex)
                    .Select(citation => citation.chunk.Text ?? ""));

                var summaryTerms = new[] { "objective", "aim", "scope", "responsibilities", "implementation", "access", "retention" };
                var snippets = summaryTerms
                    .Select(term => ExtractBestSnippet(text, new List<string> { term }, 360))
                    .Where(snippet => !string.IsNullOrWhiteSpace(snippet))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .Take(4)
                    .ToList();

                if (snippets.Count == 0)
                    snippets.Add(Preview(text, 700));

                parts.Add($"From {document.Title}:\n- {string.Join("\n- ", snippets)}");
            }

            if (parts.Count == 0)
                return "I found relevant information, but I could not summarize it reliably.";

            return "Here is a source-based summary:\n\n" + string.Join("\n\n", parts);
        }

        private static string BuildCalendarExtractiveAnswer(List<(DocumentChunk chunk, float score)> citations, string questionText)
        {
            var entries = citations
                .SelectMany(citation => ParseCalendarEntries(citation.chunk.Text))
                .GroupBy(entry => $"{entry.Date:yyyy-MM-dd}:{NormalizeForMatch(entry.Label)}")
                .Select(group => group.First())
                .OrderBy(entry => entry.Date)
                .ThenBy(entry => entry.Label)
                .ToList();

            var focusedEntries = SelectCalendarEntries(entries, questionText);
            if (focusedEntries.Count > 0)
            {
                var titleFromEntries = citations[0].chunk.Document?.Title ?? "Calendar";
                var formattedLines = FormatCalendarEntries(focusedEntries);
                var lines = formattedLines
                    .Take(12)
                    .Select(line => $"- {line}")
                    .ToList();

                var note = formattedLines.Count > lines.Count
                    ? "\n\nI found more matching dates too. Ask with a month or event name to narrow it down."
                    : "";

                return $"According to \"{titleFromEntries}\":\n\n{string.Join("\n", lines)}{note}";
            }

            var terms = ExpandQueryTerms(Tokenize(questionText))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var snippets = citations
                .SelectMany(citation => ExtractCalendarSnippets(citation.chunk.Text, terms))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Take(6)
                .ToList();

            if (snippets.Count == 0)
            {
                snippets = citations
                    .Take(2)
                    .Select(citation => NormalizeCalendarText(Preview(citation.chunk.Text, 600)))
                    .ToList();
            }

            var title = citations[0].chunk.Document?.Title ?? "Calendar";

            return $"I found calendar entries related to your question in \"{title}\":\n\n- {string.Join("\n- ", snippets)}\n\nThe calendar is extracted from a PDF table, so for exact dates ask with a specific event or month, for example: \"When are exams in May?\" or \"When does semester 2 start?\"";
        }

        private static List<CalendarEntry> ParseCalendarEntries(string text)
        {
            var value = text ?? "";
            var entries = new List<CalendarEntry>();

            var firstStart = FindCalendarPatternIndex(value, @"Ven\s*1\s*Lun\s*1");
            if (firstStart >= 0)
            {
                var firstEnd = FindEarliestIndex(value, firstStart + 10, "(*) Sous", "RDD =", "2026F");
                var firstBody = value[firstStart..firstEnd];
                entries.AddRange(ParseCalendarTable(firstBody, FirstCalendarMonths));
            }

            var secondStart = FindCalendarPatternIndex(value, @"Dim\s*1\s*Semaine\s*7\s*Mer\s*1");
            if (secondStart >= 0)
            {
                var secondEnd = FindEarliestIndex(value, secondStart + 10, "MarsF", "Approuv");
                var secondBody = value[secondStart..secondEnd];
                entries.AddRange(ParseCalendarTable(secondBody, SecondCalendarMonths));
            }

            return entries;
        }

        private static List<CalendarEntry> ParseCalendarTable(string text, IReadOnlyList<CalendarMonthSpec> months)
        {
            var entries = new List<CalendarEntry>();
            var matches = CalendarCellRegex.Matches(text ?? "");
            var currentDay = 0;
            var monthCursor = 0;

            foreach (Match match in matches)
            {
                if (!int.TryParse(match.Groups["day"].Value, out var day))
                    continue;

                if (day != currentDay)
                {
                    currentDay = day;
                    monthCursor = 0;
                }

                var dow = NormalizeFrenchWeekday(match.Groups["dow"].Value);
                var monthIndex = FindMatchingCalendarMonth(months, monthCursor, day, dow);
                if (monthIndex < 0)
                    continue;

                monthCursor = monthIndex + 1;

                var label = CleanCalendarLabel(match.Groups["label"].Value);
                if (string.IsNullOrWhiteSpace(label))
                    continue;

                var month = months[monthIndex];
                entries.Add(new CalendarEntry
                {
                    Date = new DateTime(month.Year, month.Month, day),
                    Label = label
                });
            }

            return entries;
        }

        private static List<CalendarEntry> SelectCalendarEntries(List<CalendarEntry> entries, string questionText)
        {
            if (entries.Count == 0)
                return new List<CalendarEntry>();

            var normalizedQuestion = NormalizeForMatch(questionText);
            var tokens = Tokenize(questionText)
                .Select(NormalizeForMatch)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            var months = ExtractMonthNumbers(normalizedQuestion);

            IEnumerable<CalendarEntry> query = entries;
            if (months.Count > 0)
                query = query.Where(entry => months.Contains(entry.Date.Month));

            var wantsExam = tokens.Overlaps(new[] { "exam", "exams", "examen", "examens", "examination", "examinations" });
            var wantsDiploma = tokens.Overlaps(new[] { "rdd", "diploma", "diplomas", "graduation", "ceremony", "ceremonies" });
            var wantsHoliday = tokens.Overlaps(new[] { "holiday", "holidays", "vacation", "break", "conge", "conges", "fete", "fetes", "class", "classes" });
            var wantsSummer = tokens.Overlaps(new[] { "summer", "trimester", "trim", "ete" });
            var wantsSemester = tokens.Overlaps(new[] { "semester", "semesters", "semestre", "semestres", "sem", "start", "starts", "begin", "begins", "beginning", "debut" });
            var wantsSummary = tokens.Overlaps(new[] { "summary", "summarize", "summarise", "overview" });

            if (wantsExam)
            {
                query = query.Where(entry => EntryLabelContainsAny(entry, "examen", "examens", "langues"));
            }
            else if (wantsDiploma)
            {
                query = query.Where(entry => EntryLabelContainsAny(entry, "rdd", "remise", "diplome", "diplomes"));
            }
            else if (wantsHoliday)
            {
                query = query.Where(IsHolidayEntry);
            }
            else if (wantsSummer)
            {
                query = query.Where(entry => EntryLabelContainsAny(entry, "debut trim", "trim", "ete"));
            }
            else if (wantsSemester)
            {
                query = query.Where(entry => EntryLabelContainsAny(entry, "debut sem"));

                if (normalizedQuestion.Contains("semester 1") || normalizedQuestion.Contains("semestre 1") || normalizedQuestion.Contains("sem 1"))
                    query = query.Where(entry => NormalizeForMatch(entry.Label).Contains("1"));
                else if (normalizedQuestion.Contains("semester 2") || normalizedQuestion.Contains("semestre 2") || normalizedQuestion.Contains("sem 2"))
                    query = query.Where(entry => NormalizeForMatch(entry.Label).Contains("2"));
            }
            else if (wantsSummary)
            {
                query = query.Where(IsMajorCalendarEntry);
            }
            else
            {
                var expandedTerms = ExpandQueryTerms(Tokenize(questionText))
                    .Select(NormalizeForMatch)
                    .Where(term => term.Length > 2)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                query = query.Where(entry => expandedTerms.Any(term => NormalizeForMatch(entry.Label).Contains(term)));
            }

            var result = query
                .OrderBy(entry => entry.Date)
                .ThenBy(entry => entry.Label)
                .Take(40)
                .ToList();

            if (result.Count == 0 && months.Count > 0)
            {
                result = entries
                    .Where(entry => months.Contains(entry.Date.Month))
                    .OrderBy(entry => entry.Date)
                    .ThenBy(entry => entry.Label)
                    .Take(20)
                    .ToList();
            }

            return result;
        }

        private static List<string> FormatCalendarEntries(List<CalendarEntry> entries)
        {
            var ranges = new List<CalendarRange>();

            foreach (var entry in entries.OrderBy(entry => entry.Date).ThenBy(entry => entry.Label))
            {
                var labelKey = NormalizeForMatch(entry.Label);
                var last = ranges.LastOrDefault();

                if (last != null &&
                    last.LabelKey == labelKey &&
                    entry.Date.Date == last.End.Date.AddDays(1))
                {
                    last.End = entry.Date.Date;
                    continue;
                }

                ranges.Add(new CalendarRange
                {
                    Start = entry.Date.Date,
                    End = entry.Date.Date,
                    Label = entry.Label,
                    LabelKey = labelKey
                });
            }

            return ranges
                .Select(range => $"{FormatDateRange(range.Start, range.End)}: {range.Label}")
                .ToList();
        }

        private static string FormatDateRange(DateTime start, DateTime end)
        {
            if (start.Date == end.Date)
                return start.ToString("MMMM d, yyyy", CultureInfo.InvariantCulture);

            if (start.Year == end.Year && start.Month == end.Month)
                return $"{start.ToString("MMMM", CultureInfo.InvariantCulture)} {start.Day}-{end.Day}, {start.Year}";

            if (start.Year == end.Year)
                return $"{start.ToString("MMMM d", CultureInfo.InvariantCulture)}-{end.ToString("MMMM d", CultureInfo.InvariantCulture)}, {start.Year}";

            return $"{start.ToString("MMMM d, yyyy", CultureInfo.InvariantCulture)}-{end.ToString("MMMM d, yyyy", CultureInfo.InvariantCulture)}";
        }

        private static bool IsHolidayEntry(CalendarEntry entry)
        {
            return EntryLabelContainsAny(
                entry,
                "fete",
                "noel",
                "paques",
                "fitr",
                "adha",
                "maouled",
                "hegre",
                "hegire",
                "achoura",
                "assomption",
                "toussaint",
                "nouvel an",
                "saint",
                "annonciation",
                "independance",
                "travail",
                "resist",
                "liberat",
                "conge",
                "conges");
        }

        private static bool IsMajorCalendarEntry(CalendarEntry entry)
        {
            return EntryLabelContainsAny(
                entry,
                "debut sem",
                "debut trim",
                "examen",
                "rdd",
                "fete",
                "noel",
                "paques",
                "fitr",
                "adha",
                "maouled",
                "rentree admin");
        }

        private static bool EntryLabelContainsAny(CalendarEntry entry, params string[] needles)
        {
            var label = NormalizeForMatch(entry.Label);
            return needles.Any(needle => label.Contains(NormalizeForMatch(needle)));
        }

        private static HashSet<int> ExtractMonthNumbers(string normalizedQuestion)
        {
            var months = new HashSet<int>();
            var names = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
            {
                ["january"] = 1,
                ["janvier"] = 1,
                ["february"] = 2,
                ["feb"] = 2,
                ["fevrier"] = 2,
                ["march"] = 3,
                ["mars"] = 3,
                ["april"] = 4,
                ["avril"] = 4,
                ["may"] = 5,
                ["mai"] = 5,
                ["june"] = 6,
                ["juin"] = 6,
                ["july"] = 7,
                ["juillet"] = 7,
                ["august"] = 8,
                ["aout"] = 8,
                ["september"] = 9,
                ["sept"] = 9,
                ["septembre"] = 9,
                ["october"] = 10,
                ["octobre"] = 10,
                ["november"] = 11,
                ["novembre"] = 11,
                ["december"] = 12,
                ["decembre"] = 12
            };

            foreach (var (name, number) in names)
            {
                if (Regex.IsMatch(normalizedQuestion, $@"\b{Regex.Escape(name)}\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
                    months.Add(number);
            }

            return months;
        }

        private static int FindCalendarPatternIndex(string text, string pattern)
        {
            var match = Regex.Match(text ?? "", pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            return match.Success ? match.Index : -1;
        }

        private static int FindEarliestIndex(string text, int startAt, params string[] needles)
        {
            var indexes = needles
                .Select(needle => text.IndexOf(needle, startAt, StringComparison.OrdinalIgnoreCase))
                .Where(index => index >= 0)
                .ToList();

            return indexes.Count == 0 ? text.Length : indexes.Min();
        }

        private static int FindMatchingCalendarMonth(IReadOnlyList<CalendarMonthSpec> months, int startIndex, int day, string dow)
        {
            for (var index = startIndex; index < months.Count; index++)
            {
                var month = months[index];
                if (day > DateTime.DaysInMonth(month.Year, month.Month))
                    continue;

                var date = new DateTime(month.Year, month.Month, day);
                if (GetFrenchWeekday(date) == dow)
                    return index;
            }

            for (var index = startIndex; index < months.Count; index++)
            {
                var month = months[index];
                if (day <= DateTime.DaysInMonth(month.Year, month.Month))
                    return index;
            }

            return -1;
        }

        private static string NormalizeFrenchWeekday(string dow)
        {
            var normalized = (dow ?? "").Trim().ToLowerInvariant();
            if (normalized.StartsWith("lun"))
                return "Lun";
            if (normalized.StartsWith("mar"))
                return "Mar";
            if (normalized.StartsWith("mer"))
                return "Mer";
            if (normalized.StartsWith("jeu"))
                return "Jeu";
            if (normalized.StartsWith("ven"))
                return "Ven";
            if (normalized.StartsWith("sam"))
                return "Sam";
            if (normalized.StartsWith("dim"))
                return "Dim";

            return normalized;
        }

        private static string GetFrenchWeekday(DateTime date)
        {
            return date.DayOfWeek switch
            {
                DayOfWeek.Monday => "Lun",
                DayOfWeek.Tuesday => "Mar",
                DayOfWeek.Wednesday => "Mer",
                DayOfWeek.Thursday => "Jeu",
                DayOfWeek.Friday => "Ven",
                DayOfWeek.Saturday => "Sam",
                DayOfWeek.Sunday => "Dim",
                _ => ""
            };
        }

        private static string CleanCalendarLabel(string text)
        {
            var cleaned = NormalizeCalendarText(text);
            cleaned = Regex.Replace(cleaned, @"\s+", " ").Trim();
            return cleaned.Trim('-', ';', ',', '.');
        }

        private static List<string> ExtractCalendarSnippets(string text, List<string> terms)
        {
            var snippets = new List<string>();
            var normalized = NormalizeCalendarText(text);
            var lower = normalized.ToLowerInvariant();

            foreach (var term in terms)
            {
                var normalizedTerm = NormalizeForMatch(term);
                if (normalizedTerm.Length < 3)
                    continue;

                var searchTerms = new[] { normalizedTerm, NormalizeForMatch(RemoveAccents(term)) }
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                foreach (var searchTerm in searchTerms)
                {
                    var index = lower.IndexOf(searchTerm, StringComparison.OrdinalIgnoreCase);
                    while (index >= 0 && snippets.Count < 8)
                    {
                        var start = Math.Max(0, index - 90);
                        var length = Math.Min(220, normalized.Length - start);
                        snippets.Add(normalized.Substring(start, length).Trim());
                        index = lower.IndexOf(searchTerm, index + searchTerm.Length, StringComparison.OrdinalIgnoreCase);
                    }
                }
            }

            return snippets;
        }

        private static string NormalizeCalendarText(string text)
        {
            var value = text ?? "";

            value = Regex.Replace(
                value,
                @"(?<!^)(?=(Vend|Lun|Mar|Mer|Jeu|Ven|Sam|Dim)\s*\d{1,2})",
                " ",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

            value = Regex.Replace(
                value,
                @"(Vend|Lun|Mar|Mer|Jeu|Ven|Sam|Dim)\s*(\d{1,2})(?=\p{L})",
                "$1$2 ",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

            value = Regex.Replace(
                value,
                @"(?<=\p{Ll})(?=\p{Lu})",
                " ",
                RegexOptions.CultureInvariant);

            return string.Join(" ", value.Split(default(string[]), StringSplitOptions.RemoveEmptyEntries));
        }

        private static string ExtractBestSnippet(string text, List<string> terms, int maxLength)
        {
            var cleaned = string.Join(" ", (text ?? "").Split(default(string[]), StringSplitOptions.RemoveEmptyEntries));
            if (terms.Count == 0)
                return Preview(cleaned, maxLength);

            var sectionStart = FindSectionStart(cleaned, terms);
            if (sectionStart >= 0)
            {
                var sectionLength = Math.Min(maxLength, cleaned.Length - sectionStart);
                var sectionSnippet = cleaned.Substring(sectionStart, sectionLength).Trim();

                if (sectionStart + sectionLength < cleaned.Length)
                    sectionSnippet += "...";

                return PolishSnippet(sectionSnippet);
            }

            var lower = cleaned.ToLowerInvariant();
            var indexes = terms
                .Select(term => lower.IndexOf(term.ToLowerInvariant(), StringComparison.OrdinalIgnoreCase))
                .Where(index => index >= 0)
                .ToList();

            if (indexes.Count == 0)
                return Preview(cleaned, maxLength);

            var first = indexes.Min();
            var start = Math.Max(0, first - 180);
            var length = Math.Min(maxLength, cleaned.Length - start);
            var snippet = cleaned.Substring(start, length).Trim();

            if (start > 0)
                snippet = "..." + snippet;

            if (start + length < cleaned.Length)
                snippet += "...";

            return PolishSnippet(snippet);
        }

        private static int FindSectionStart(string text, List<string> terms)
        {
            var headingMap = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["objective"] = new[] { "OBJECTIVE" },
                ["aim"] = new[] { "AIM" },
                ["scope"] = new[] { "SCOPE" },
                ["responsibility"] = new[] { "RESPONSIBILITIES", "RESPONSIBILITY" },
                ["responsibilities"] = new[] { "RESPONSIBILITIES", "RESPONSIBILITY" },
                ["rule"] = new[] { "CCTV RULES", "RULES AND REGULATIONS", "RULES" },
                ["rules"] = new[] { "CCTV RULES", "RULES AND REGULATIONS", "RULES" },
                ["implementation"] = new[] { "CCTV IMPLEMENTATION", "IMPLEMENTATION" },
                ["access"] = new[] { "ACCESS REQUESTS", "ACCESS FORM" },
                ["requests"] = new[] { "ACCESS REQUESTS", "ACCESS FORM" },
                ["retention"] = new[] { "RETENTION" }
            };

            foreach (var term in terms.Select(NormalizeForMatch))
            {
                if (!headingMap.TryGetValue(term, out var headings))
                    continue;

                foreach (var heading in headings)
                {
                    var pattern = $@"\b\d+(?:\.\d+)?\.\s*{Regex.Escape(heading)}(?=[A-Z0-9\s])";
                    var matches = Regex.Matches(text, pattern, RegexOptions.CultureInvariant);

                    if (matches.Count > 0)
                        return matches[^1].Index;
                }
            }

            return -1;
        }

        private static string PolishSnippet(string text)
        {
            var value = text ?? "";
            value = Regex.Replace(
                value,
                @"\b\d([1-7])\.\s+(OBJECTIVE|AIM|SCOPE|RESPONSIBILITIES|CCTV|POLICY|ANNEX)",
                "$1. $2",
                RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=[A-Z])(?=This\b)", " ", RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=[A-Z])(?=\d+\.\d+)", " ", RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=[a-z)])(?=\d+(?:\.\d+)?\.\s*[A-Z])", " ", RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=\p{Ll})(?=\p{Lu})", " ", RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"\s+", " ", RegexOptions.CultureInvariant).Trim();
            return value;
        }

        private static string Preview(string text, int maxLength)
        {
            var cleaned = string.Join(" ", (text ?? "").Split(default(string[]), StringSplitOptions.RemoveEmptyEntries));
            if (cleaned.Length <= maxLength)
                return cleaned;

            return cleaned[..maxLength].TrimEnd() + "...";
        }

        private static float ScoreChunk(DocumentChunk chunk, List<string> terms, string questionText)
        {
            var text = NormalizeForMatch(chunk.Text);
            var title = NormalizeForMatch(chunk.Document?.Title);
            var metadata = NormalizeForMatch($"{chunk.Document?.Department} {chunk.Document?.Course} {chunk.Document?.Category} {chunk.Document?.Tags}");
            var question = NormalizeForMatch(questionText);

            float score = 0;

            if (question.Length > 8 && text.Contains(question))
                score += 8;

            foreach (var term in terms)
            {
                var normalizedTerm = NormalizeForMatch(term);
                if (normalizedTerm.Length <= 2)
                    continue;

                if (text.Contains(normalizedTerm))
                    score += 1;

                if (title.Contains(normalizedTerm))
                    score += 2;

                if (metadata.Contains(normalizedTerm))
                    score += 1.5f;
            }

            return score;
        }

        private static List<string> Tokenize(string text)
        {
            var tokens = new List<string>();
            var current = new List<char>();

            foreach (var ch in (text ?? "").ToLowerInvariant())
            {
                if (char.IsLetterOrDigit(ch))
                {
                    current.Add(ch);
                }
                else if (current.Count > 0)
                {
                    AddToken(tokens, current);
                    current.Clear();
                }
            }

            if (current.Count > 0)
                AddToken(tokens, current);

            return tokens;
        }

        private static void AddToken(List<string> tokens, List<char> current)
        {
            var token = new string(current.ToArray());
            if (token.Length > 2 && !StopWords.Contains(token))
                tokens.Add(token);
        }

        private static IEnumerable<string> ExpandQueryTerms(IEnumerable<string> tokens)
        {
            var expanded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var rawToken in tokens)
            {
                var token = NormalizeForMatch(rawToken);
                if (token.Length <= 2)
                    continue;

                AddTerms(expanded, token);

                switch (token)
                {
                    case "calendar":
                    case "calendrier":
                    case "deadline":
                    case "deadlines":
                        AddTerms(expanded, "calendar", "calendrier", "date", "dates");
                        break;

                    case "exam":
                    case "exams":
                    case "examination":
                    case "examinations":
                    case "examen":
                    case "examens":
                        AddTerms(expanded, "exam", "exams", "examination", "examinations", "examen", "examens");
                        break;

                    case "semester":
                    case "semesters":
                    case "semestre":
                    case "semestres":
                    case "sem":
                        AddTerms(expanded, "semester", "semesters", "semestre", "semestres", "sem");
                        break;

                    case "start":
                    case "starts":
                    case "begin":
                    case "begins":
                    case "beginning":
                    case "debut":
                        AddTerms(expanded, "start", "begin", "beginning", "debut");
                        break;

                    case "holiday":
                    case "holidays":
                    case "vacation":
                    case "break":
                    case "conge":
                    case "conges":
                    case "fete":
                    case "fetes":
                        AddTerms(expanded, "holiday", "holidays", "vacation", "break", "conge", "conges", "fete", "fetes", "noel", "paques");
                        break;

                    case "class":
                    case "classes":
                    case "course":
                    case "cours":
                        AddTerms(expanded, "class", "classes", "course", "cours");
                        break;

                    case "graduation":
                    case "diploma":
                    case "diplomas":
                    case "ceremony":
                    case "ceremonies":
                    case "rdd":
                        AddTerms(expanded, "graduation", "diploma", "diplomas", "ceremony", "ceremonies", "rdd", "remise", "diplome", "diplomes");
                        break;

                    case "summer":
                    case "trimester":
                    case "trim":
                    case "ete":
                        AddTerms(expanded, "summer", "trimester", "trim", "ete");
                        break;

                    case "christmas":
                    case "noel":
                        AddTerms(expanded, "christmas", "noel");
                        break;

                    case "easter":
                    case "paques":
                        AddTerms(expanded, "easter", "paques");
                        break;

                    case "independence":
                    case "independance":
                        AddTerms(expanded, "independence", "independance");
                        break;

                    case "labor":
                    case "labour":
                    case "work":
                    case "travail":
                        AddTerms(expanded, "labor", "labour", "work", "travail");
                        break;
                }
            }

            return expanded;
        }

        private static void AddTerms(HashSet<string> terms, params string[] values)
        {
            foreach (var value in values)
            {
                var normalized = NormalizeForMatch(value);
                if (normalized.Length > 2)
                    terms.Add(normalized);
            }
        }

        private static string NormalizeForMatch(string? text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return "";

            var withoutAccents = RemoveAccents(text).ToLowerInvariant();
            var builder = new StringBuilder(withoutAccents.Length);
            var previousWasSpace = true;

            foreach (var ch in withoutAccents)
            {
                if (char.IsLetterOrDigit(ch))
                {
                    builder.Append(ch);
                    previousWasSpace = false;
                }
                else if (!previousWasSpace)
                {
                    builder.Append(' ');
                    previousWasSpace = true;
                }
            }

            return builder.ToString().Trim();
        }

        private static string RemoveAccents(string text)
        {
            var normalized = (text ?? "").Normalize(NormalizationForm.FormD);
            var builder = new StringBuilder(normalized.Length);

            foreach (var ch in normalized)
            {
                if (CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
                    builder.Append(ch);
            }

            return builder.ToString().Normalize(NormalizationForm.FormC);
        }

        private static readonly Regex CalendarCellRegex = new(
            @"(?<dow>Vend|Lun|Mar|Mer|Jeu|Ven|Sam|Dim)\s*(?<day>\d{1,2})(?<label>.*?)(?=(Vend|Lun|Mar|Mer|Jeu|Ven|Sam|Dim)\s*\d{1,2}|$)",
            RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.CultureInvariant);

        private static readonly CalendarMonthSpec[] FirstCalendarMonths =
        {
            new(2025, 8, "August"),
            new(2025, 9, "September"),
            new(2025, 10, "October"),
            new(2025, 11, "November"),
            new(2025, 12, "December"),
            new(2026, 1, "January"),
            new(2026, 2, "February")
        };

        private static readonly CalendarMonthSpec[] SecondCalendarMonths =
        {
            new(2026, 3, "March"),
            new(2026, 4, "April"),
            new(2026, 5, "May"),
            new(2026, 6, "June"),
            new(2026, 7, "July"),
            new(2026, 8, "August"),
            new(2026, 9, "September")
        };

        private sealed record CalendarMonthSpec(int Year, int Month, string Name);

        private sealed class CalendarEntry
        {
            public DateTime Date { get; init; }
            public string Label { get; init; } = "";
        }

        private sealed class CalendarRange
        {
            public DateTime Start { get; init; }
            public DateTime End { get; set; }
            public string Label { get; init; } = "";
            public string LabelKey { get; init; } = "";
        }

        private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
        {
            "the", "and", "for", "are", "you", "your", "with", "that", "this", "what", "when",
            "where", "who", "why", "how", "can", "could", "should", "would", "about", "into",
            "from", "does", "have", "has", "had", "will", "shall", "may", "might", "must",
            "please", "tell", "explain", "give", "show", "document", "documents", "is", "was",
            "were", "am", "a", "an", "of", "to", "in", "on", "at", "by", "me", "my", "our",
            "their", "it", "its", "as", "or"
        };

        private class GenerateRequest
        {
            [JsonPropertyName("question")]
            public string Question { get; set; } = "";

            [JsonPropertyName("chunks")]
            public List<GenerateChunkItem> Chunks { get; set; } = new();
        }

        private class GenerateChunkItem
        {
            [JsonPropertyName("chunk_id")]
            public int ChunkId { get; set; }

            [JsonPropertyName("document_title")]
            public string DocumentTitle { get; set; } = "";

            [JsonPropertyName("text")]
            public string Text { get; set; } = "";

            [JsonPropertyName("score")]
            public float Score { get; set; }
        }

        private class GenerateResponse
        {
            [JsonPropertyName("answer")]
            public string Answer { get; set; } = "";

            [JsonPropertyName("supported")]
            public bool Supported { get; set; }

            [JsonPropertyName("used_chunk_ids")]
            public List<int> UsedChunkIds { get; set; } = new();
        }
    }
}
