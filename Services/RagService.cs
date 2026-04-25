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
                SessionId = "student-chat",
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
            else if (explicitDocumentMatches.Count == 0 && LooksLikeCalendarQuestion(questionText))
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

            if (TryGetFollowUpTransformInstruction(questionText, out var transformInstruction))
            {
                var previousTurn = await GetPreviousAnsweredTurnAsync(userId, question.Id);
                if (previousTurn == null)
                {
                    var noContextAnswer = await SaveAnswerAsync(question.Id, "I do not have a previous answer to rewrite yet. Ask me a document question first, then I can translate, shorten, or explain that answer.", 0.3f);
                    return (noContextAnswer, new List<(DocumentChunk chunk, float score)>());
                }

                var transformedText = await TransformPreviousAnswerAsync(transformInstruction, previousTurn.AnswerText);
                var transformedAnswer = await SaveAnswerAsync(question.Id, transformedText, previousTurn.Citations.Count > 0 ? previousTurn.Citations[0].score : 0.8f);

                foreach (var citation in previousTurn.Citations)
                {
                    _db.ChunkUsages.Add(new ChunkUsage
                    {
                        AnswerId = transformedAnswer.Id,
                        ChunkId = citation.chunk.Id,
                        Score = citation.score
                    });
                }

                await _db.SaveChangesAsync();
                return (transformedAnswer, previousTurn.Citations);
            }

            var topicClarification = TryBuildTopicClarification(questionText, documents, explicitDocumentMatches, targetDocument);
            if (!string.IsNullOrWhiteSpace(topicClarification))
            {
                var clarificationAnswer = await SaveAnswerAsync(question.Id, topicClarification, 0.95f);
                return (clarificationAnswer, new List<(DocumentChunk chunk, float score)>());
            }

            int topK = int.Parse(_cfg["Rag:TopK"] ?? "4");
            var searchTopK = targetDocument == null ? topK : Math.Max(topK, 8);
            var citations = await TryVectorSearchAsync(questionText, searchTopK, targetDocument?.Id);

            if (citations.Count == 0)
                citations = await SearchDatabaseChunksAsync(questionText, searchTopK, targetDocument?.Id);

            string finalAnswer = HumanizeAnswerText(await GenerateAnswerAsync(questionText, citations));
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
                    var embedding = await _ml.EmbedAsync(chunkRow.Text, "document");

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
                var questionEmbedding = await _ml.EmbedAsync(questionText, "query");
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

            var aiRanked = await TryAiRerankChunksAsync(questionText, candidates, topK);
            if (aiRanked.Count > 0)
                return aiRanked;

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

        private async Task<List<(DocumentChunk chunk, float score)>> TryAiRerankChunksAsync(string questionText, List<DocumentChunk> candidates, int topK)
        {
            if (candidates.Count == 0)
                return new List<(DocumentChunk chunk, float score)>();

            var aiBaseUrl = (_cfg["AiService:BaseUrl"] ?? _cfg["MlService:BaseUrl"] ?? "http://127.0.0.1:8000").TrimEnd('/');
            var request = new RerankRequest
            {
                Question = questionText,
                Chunks = candidates
                    .Select(chunk => new RerankChunkItem
                    {
                        ChunkId = chunk.Id,
                        DocumentTitle = chunk.Document?.Title ?? "Unknown Document",
                        Text = Preview(chunk.Text, 1400),
                        Score = 0
                    })
                    .ToList()
            };

            try
            {
                var response = await _http.PostAsJsonAsync($"{aiBaseUrl}/rerank", request);
                if (!response.IsSuccessStatusCode)
                    return new List<(DocumentChunk chunk, float score)>();

                var result = await response.Content.ReadFromJsonAsync<RerankResponse>();
                if (result?.Results == null || result.Results.Count == 0)
                    return new List<(DocumentChunk chunk, float score)>();

                var chunkById = candidates.ToDictionary(chunk => chunk.Id);

                return result.Results
                    .Where(item => chunkById.ContainsKey(item.ChunkId))
                    .Take(topK)
                    .Select(item => (chunkById[item.ChunkId], item.Score))
                    .ToList();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "AI rerank failed. Falling back to keyword database search.");
                return new List<(DocumentChunk chunk, float score)>();
            }
        }

        private async Task<Answer> SaveAnswerAsync(int questionId, string text, float confidence)
        {
            var answer = new Answer
            {
                QuestionId = questionId,
                Text = HumanizeAnswerText(text),
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

        private async Task<PreviousTurn?> GetPreviousAnsweredTurnAsync(int userId, int currentQuestionId)
        {
            var previous = await _db.Questions
                .Include(q => q.Answer)
                    .ThenInclude(a => a!.ChunkUsages)
                        .ThenInclude(cu => cu.Chunk)
                            .ThenInclude(c => c!.Document)
                .Where(q => q.UserId == userId && q.Id != currentQuestionId && q.Answer != null)
                .OrderByDescending(q => q.Id)
                .FirstOrDefaultAsync();

            if (previous?.Answer == null)
                return null;

            var citations = previous.Answer.ChunkUsages
                .Where(cu => cu.Chunk != null)
                .Select(cu => (cu.Chunk!, cu.Score))
                .ToList();

            return new PreviousTurn(previous.Text, previous.Answer.Text, citations);
        }

        private static bool TryGetFollowUpTransformInstruction(string questionText, out string instruction)
        {
            var normalized = NormalizeForMatch(questionText);
            instruction = "";

            var refersToPrevious = QuestionContainsAny(normalized, "it", "them", "that", "this", "previous", "answer", "above");
            var asksTranslate = QuestionContainsAny(normalized, "translate", "translation", "english", "anglais", "french", "francais");
            var asksRewrite = QuestionContainsAny(normalized, "shorter", "simplify", "simpler", "summarize", "summarise", "explain", "rewrite", "rephrase");

            if (!refersToPrevious || (!asksTranslate && !asksRewrite))
                return false;

            if (QuestionContainsAny(normalized, "english", "anglais"))
                instruction = "Translate the previous answer to clear English.";
            else if (QuestionContainsAny(normalized, "french", "francais"))
                instruction = "Translate the previous answer to clear French.";
            else if (QuestionContainsAny(normalized, "shorter", "summarize", "summarise"))
                instruction = "Make the previous answer shorter while keeping the important facts.";
            else if (QuestionContainsAny(normalized, "simplify", "simpler", "explain"))
                instruction = "Explain the previous answer in simpler words.";
            else
                instruction = "Rewrite the previous answer clearly.";

            return true;
        }

        private async Task<string> TransformPreviousAnswerAsync(string instruction, string previousAnswer)
        {
            if (NormalizeForMatch(instruction).Contains("english"))
            {
                var offlineTranslation = TranslateAcademicAnswerToEnglish(previousAnswer);
                if (!string.Equals(offlineTranslation, previousAnswer, StringComparison.Ordinal))
                    return offlineTranslation;
            }

            var aiBaseUrl = (_cfg["AiService:BaseUrl"] ?? _cfg["MlService:BaseUrl"] ?? "http://127.0.0.1:8000").TrimEnd('/');

            try
            {
                var response = await _http.PostAsJsonAsync($"{aiBaseUrl}/transform", new TransformRequest
                {
                    Instruction = instruction,
                    Text = previousAnswer
                });

                if (!response.IsSuccessStatusCode)
                    return previousAnswer;

                var result = await response.Content.ReadFromJsonAsync<TransformResponse>();
                var transformed = HumanizeAnswerText(result?.Answer ?? previousAnswer);

                if (NormalizeForMatch(instruction).Contains("english") && LooksFrenchLike(transformed))
                    return TranslateAcademicAnswerToEnglish(previousAnswer);

                return transformed;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "AI transform failed. Returning the previous answer unchanged.");
                return previousAnswer;
            }
        }

        private static string TranslateAcademicAnswerToEnglish(string answer)
        {
            var lines = (answer ?? "").Split('\n');
            var changed = false;
            var translated = new List<string>();

            foreach (var line in lines)
            {
                if (!line.TrimStart().StartsWith("- ", StringComparison.Ordinal))
                {
                    translated.Add(line);
                    continue;
                }

                var separator = line.IndexOf(':');
                if (separator < 0)
                {
                    translated.Add(line);
                    continue;
                }

                var label = line[..(separator + 1)];
                var content = line[(separator + 1)..].Trim();
                var translatedContent = TranslateAcademicSentenceToEnglish(content);
                changed = changed || !string.Equals(content, translatedContent, StringComparison.Ordinal);
                translated.Add($"{label} {translatedContent}");
            }

            var result = string.Join("\n", translated);

            if (!changed && LooksFrenchLike(answer ?? ""))
                return (answer ?? "").Replace("The main points in", "The main points in", StringComparison.Ordinal);

            return result;
        }

        private static bool LooksFrenchLike(string text)
        {
            var normalized = NormalizeForMatch(text);
            return QuestionContainsAny(
                normalized,
                "etudiant",
                "etudiants",
                "inscription",
                "examen",
                "diplome",
                "credits",
                "enseignement",
                "annee",
                "semestre");
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

            if (LooksLikeCalendarQuestion(questionText) && recentDocument != null && IsCalendarDocument(recentDocument))
                return null;

            if (LooksLikeCalendarQuestion(questionText) && explicitDocumentMatches.Any(IsCalendarDocument))
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
                   normalized.Contains("overview") ||
                   normalized.Contains("main ") ||
                   normalized.Contains("key ") ||
                   normalized.Contains("important ") ||
                   normalized.Contains("regulation") ||
                   normalized.Contains("regulations") ||
                   normalized.Contains("rules");
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
                    "semester end dates",
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

            var lower = NormalizeForMatch(text);
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
            AddTopicIfPresent(topics, lower, "ethique", "ethics and obligations");
            AddTopicIfPresent(topics, lower, "election", "elections");
            AddTopicIfPresent(topics, lower, "scrutin", "voting rules");
            AddTopicIfPresent(topics, lower, "committee", "committee");
            AddTopicIfPresent(topics, lower, "bureau", "bureau composition");
            AddTopicIfPresent(topics, lower, "membership", "membership");
            AddTopicIfPresent(topics, lower, "membre", "membership");
            AddTopicIfPresent(topics, lower, "quorum", "quorum");
            AddTopicIfPresent(topics, lower, "mandat", "mandate duration");
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
                "semester end dates" => $"When do the semesters end in {title}?",
                "holidays and no-class days" => $"What holidays or no-class days are listed in {title}?",
                "diploma ceremonies" => $"When are the diploma ceremonies in {title}?",
                "summer trimester" => $"When does the summer trimester start in {title}?",
                "objective" => $"What is the objective of {title}?",
                "aim" => $"What is the aim of {title}?",
                "scope" => $"What is the scope of {title}?",
                "responsibilities" => $"Who is responsible for {title}?",
                "retention period" => $"What is the retention period in {title}?",
                "access requests" => $"How are access requests handled in {title}?",
                "ethics and obligations" => $"What obligations does {title} set for students?",
                "elections" => $"How do elections work in {title}?",
                "voting rules" => $"What are the voting rules in {title}?",
                "bureau composition" => $"How is the bureau composed in {title}?",
                "membership" => $"Who is considered a member in {title}?",
                "quorum" => $"What quorum is required in {title}?",
                "mandate duration" => $"How long is the mandate in {title}?",
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
                "semester", "semesters", "semestre", "semestres", "sem", "holiday", "holidays",
                "vacation", "break", "rdd", "diploma", "diplomas", "graduation",
                "summer", "trimester", "trim", "end", "ends", "finish", "finishes",
                "finished", "ending", "fin"
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

            var sourceAnswer = BuildExtractiveAnswer(citations, questionText);
            var useGenerator = bool.TryParse(_cfg["Rag:UseGenerator"], out var configuredUseGenerator) && configuredUseGenerator;
            if (!useGenerator)
                return sourceAnswer;

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
                    return sourceAnswer;
                }

                var result = await response.Content.ReadFromJsonAsync<GenerateResponse>();

                if (result == null || string.IsNullOrWhiteSpace(result.Answer) || !result.Supported)
                    return sourceAnswer;

                return result.Answer.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "AI generate service failed. Returning extractive answer.");
                return sourceAnswer;
            }
        }

        private static string BuildExtractiveAnswer(List<(DocumentChunk chunk, float score)> citations, string questionText)
        {
            if (citations.Count > 0 && citations.All(c => c.chunk.Document != null && IsCalendarDocument(c.chunk.Document)))
                return BuildCalendarExtractiveAnswer(citations, questionText);

            if (IsSummaryLike(questionText))
                return BuildSummaryAnswer(citations, questionText);

            var terms = ExpandQueryTerms(Tokenize(questionText))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var grouped = citations
                .Where(citation => citation.chunk.Document != null)
                .GroupBy(citation => citation.chunk.Document!.Id)
                .Take(3)
                .ToList();

            if (grouped.Count == 0)
                return "I found relevant information, but I could not format a reliable source answer.";

            var parts = new List<string>();
            foreach (var group in grouped)
            {
                var document = group.First().chunk.Document!;
                var maxSnippets = GetAnswerSnippetLimit(questionText, grouped.Count);
                var snippets = OrderCitationsForAnswer(group, terms, questionText)
                    .Select(citation =>
                    {
                        var answerTerms = FilterAnswerTerms(terms, citation.chunk.Document, questionText);
                        return ExtractBestSnippet(citation.chunk.Text, answerTerms, 680);
                    })
                    .Where(snippet => !string.IsNullOrWhiteSpace(snippet))
                    .GroupBy(snippet => NormalizeForMatch(Preview(snippet, 220)))
                    .Select(grouping => grouping.First())
                    .Take(maxSnippets)
                    .ToList();

                if (snippets.Count == 0)
                    snippets.Add(FormatSourcePreview(group.First().chunk.Text, 680));

                if (grouped.Count == 1)
                    return $"Based on \"{document.Title}\":\n\n- {string.Join("\n- ", snippets)}\n\nYou can open the proof to see the exact PDF evidence I used.";

                parts.Add($"From \"{document.Title}\":\n- {string.Join("\n- ", snippets)}");
            }

            return "I found relevant information in these uploaded documents:\n\n" + string.Join("\n\n", parts) + "\n\nYou can open the proof to see the exact PDF evidence I used.";
        }

        private static int GetAnswerSnippetLimit(string questionText, int documentGroupCount)
        {
            if (documentGroupCount > 1)
                return 2;

            var normalizedQuestion = NormalizeForMatch(questionText);
            if (QuestionContainsAny(normalizedQuestion, "election", "elections", "vote", "voting", "scrutin"))
                return 2;

            if (QuestionContainsAny(
                normalizedQuestion,
                "quorum",
                "retention",
                "retained",
                "access",
                "request",
                "requests",
                "scope",
                "objective",
                "aim",
                "mandate",
                "mandat"))
            {
                return 1;
            }

            return 3;
        }

        private static List<string> FilterAnswerTerms(List<string> terms, Document? document, string questionText)
        {
            var normalizedQuestion = NormalizeForMatch(questionText);
            var titleTerms = Tokenize(document?.Title ?? "")
                .Select(NormalizeForMatch)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            var metadataTerms = Tokenize($"{document?.Category} {document?.Department} {document?.Course}")
                .Select(NormalizeForMatch)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            var filtered = terms
                .Select(NormalizeForMatch)
                .Where(term => term.Length > 2)
                .Where(term => !titleTerms.Contains(term))
                .Where(term => !metadataTerms.Contains(term))
                .Where(term => !GenericAnswerTerms.Contains(term))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (QuestionContainsAny(normalizedQuestion, "retention", "retained", "retain"))
                return new List<string> { "retention", "retained", "period", "months", "deleted" };

            if (QuestionContainsAny(normalizedQuestion, "access", "request", "requests"))
                return new List<string> { "access requests", "access", "request", "requests", "dpo", "approved" };

            if (QuestionContainsAny(normalizedQuestion, "election", "elections", "vote", "voting", "scrutin"))
                return new List<string> { "bureau electoral", "elections", "vote", "scrutin", "candidats", "depouillement" };

            if (QuestionContainsAny(normalizedQuestion, "quorum"))
                return new List<string> { "quorum requis", "quorum", "tiers", "majorite absolue" };

            return filtered.Count == 0
                ? terms.Select(NormalizeForMatch).Where(term => term.Length > 2).Distinct(StringComparer.OrdinalIgnoreCase).ToList()
                : filtered;
        }

        private static void AddTermsToList(List<string> terms, params string[] values)
        {
            foreach (var value in values.Select(NormalizeForMatch))
            {
                if (value.Length > 2 && !terms.Contains(value, StringComparer.OrdinalIgnoreCase))
                    terms.Add(value);
            }
        }

        private static IEnumerable<(DocumentChunk chunk, float score)> OrderCitationsForAnswer(
            IEnumerable<(DocumentChunk chunk, float score)> citations,
            List<string> terms,
            string questionText)
        {
            var list = citations.ToList();
            var normalizedQuestion = NormalizeForMatch(questionText);
            var documentTitle = NormalizeForMatch(list.FirstOrDefault().chunk?.Document?.Title);

            if (QuestionContainsAny(normalizedQuestion, "quorum"))
            {
                return list
                    .OrderByDescending(citation => NormalizeForMatch(citation.chunk.Text).Contains("article 8 quorum") ? 1 : 0)
                    .ThenBy(citation => citation.chunk.ChunkIndex)
                    .ThenByDescending(citation => citation.score);
            }

            if (QuestionContainsAny(normalizedQuestion, "election", "elections", "vote", "voting", "scrutin") && documentTitle.Contains("amicale"))
            {
                return list
                    .OrderByDescending(citation => IsElectionProcedureChunk(citation.chunk.Text))
                    .ThenBy(citation => citation.chunk.ChunkIndex < 10 ? 100 + citation.chunk.ChunkIndex : citation.chunk.ChunkIndex)
                    .ThenByDescending(citation => citation.score);
            }

            return list
                .OrderByDescending(citation => citation.score)
                .ThenBy(citation => citation.chunk.ChunkIndex);
        }

        private static bool IsElectionProcedureChunk(string text)
        {
            var normalized = NormalizeForMatch(text);
            return normalized.Contains("bureau electoral") ||
                   normalized.Contains("listes des electeurs") ||
                   normalized.Contains("eligibilite des candidats") ||
                   normalized.Contains("campagne electorale") ||
                   normalized.Contains("ouverture et cloture du scrutin") ||
                   normalized.Contains("depouillement") ||
                   normalized.Contains("vote est secret");
        }

        private static bool QuestionContainsAny(string normalizedQuestion, params string[] terms)
        {
            return terms.Any(term => Regex.IsMatch(
                normalizedQuestion,
                $@"\b{Regex.Escape(NormalizeForMatch(term))}\b",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
        }

        private static string BuildSummaryAnswer(List<(DocumentChunk chunk, float score)> citations, string questionText)
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

                var snippets = TryBuildRegulationSummaryBullets(text, questionText);
                if (snippets.Count == 0)
                    snippets = ExtractSummaryBullets(text, questionText);

                snippets = snippets
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .Take(5)
                    .ToList();

                if (snippets.Count == 0)
                    snippets.Add(Preview(CleanTextForAnswer(text), 420));

                if (documentGroups.Count == 1)
                    return $"The main points in \"{document.Title}\" are these:\n\n- {string.Join("\n- ", snippets)}\n\nYou can open the proof to see the exact PDF evidence I used.";

                parts.Add($"From \"{document.Title}\":\n- {string.Join("\n- ", snippets)}");
            }

            if (parts.Count == 0)
                return "I found relevant information, but I could not summarize it reliably.";

            return "The main points from the uploaded documents are these:\n\n" + string.Join("\n\n", parts) + "\n\nYou can open the proof to see the exact PDF evidence I used.";
        }

        private static List<string> ExtractSummaryBullets(string text, string questionText)
        {
            var cleaned = CleanTextForAnswer(text);
            if (string.IsNullOrWhiteSpace(cleaned))
                return new List<string>();

            var normalizedQuestion = NormalizeForMatch(questionText);
            var terms = ExpandQueryTerms(Tokenize(questionText))
                .Select(NormalizeForMatch)
                .Where(term => term.Length > 2)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            if (QuestionContainsAny(normalizedQuestion, "study", "studies", "academic", "regulation", "regulations", "rules"))
            {
                AddTerms(terms,
                    "academic", "annee", "year", "semester", "semestre", "trimestre", "credit", "credits", "ects",
                    "inscription", "registration", "ue", "unite", "enseignement", "calendar", "calendrier",
                    "examens", "exams", "evaluation", "grade", "note", "absence", "desister", "withdraw",
                    "stage", "diplome", "cycle");
            }

            AddTerms(terms,
                "objective", "aim", "scope", "responsibilities", "implementation", "access", "retention",
                "objet", "obligations", "ethique", "amicale", "bureau", "elections", "quorum", "mandat");

            var candidates = Regex
                .Split(cleaned, @"(?<=[.!?])\s+(?=[A-Z0-9À-ÖØ-Þ])", RegexOptions.CultureInvariant)
                .Select((sentence, index) => new
                {
                    Index = index,
                    Text = PolishSummarySentence(sentence),
                    Score = ScoreSummarySentence(sentence, terms)
                })
                .Where(item => item.Score > 0 && item.Text.Length >= 45)
                .GroupBy(item => NormalizeForMatch(Preview(item.Text, 180)))
                .Select(group => group.OrderByDescending(item => item.Score).First())
                .OrderByDescending(item => item.Score)
                .ThenBy(item => item.Index)
                .Take(5)
                .OrderBy(item => item.Index)
                .Select(item => item.Text)
                .ToList();

            return candidates;
        }

        private static List<string> TryBuildRegulationSummaryBullets(string text, string questionText)
        {
            var normalizedQuestion = NormalizeForMatch(questionText);
            if (!QuestionContainsAny(normalizedQuestion, "study", "studies", "academic", "regulation", "regulations", "rules"))
                return new List<string>();

            var cleaned = CleanTextForAnswer(text);
            var sentences = Regex
                .Split(cleaned, @"(?<=[.!?])\s+(?=[A-Z0-9À-ÖØ-Þ])", RegexOptions.CultureInvariant)
                .Select(PolishSummarySentence)
                .Where(sentence => sentence.Length >= 45)
                .ToList();

            var themes = new (string Label, string[] Terms)[]
            {
                ("Academic year", new[] { "annee academique", "semestres universitaires", "trimestre ete", "calendrier" }),
                ("Credits and workload", new[] { "credits ects", "credit ects", "30 credits", "24 credits", "charge de travail" }),
                ("Registration", new[] { "inscription", "modifier leur inscription", "premiere inscription", "documents requis" }),
                ("Courses and exams", new[] { "ue", "examens finaux", "evaluation", "note", "absence" }),
                ("Diploma requirements", new[] { "diplome", "valider", "cycle", "formation generale" })
            };

            var bullets = new List<string>();
            var used = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var (label, terms) in themes)
            {
                var sentence = FindBestThemeSentence(sentences, terms, used);
                if (string.IsNullOrWhiteSpace(sentence))
                    continue;

                used.Add(NormalizeForMatch(sentence));
                bullets.Add($"{label}: {TranslateAcademicSentenceToEnglish(sentence)}");
            }

            return bullets;
        }

        private static string? FindBestThemeSentence(List<string> sentences, string[] themeTerms, HashSet<string> used)
        {
            return sentences
                .Select((sentence, index) => new
                {
                    Index = index,
                    Sentence = sentence,
                    Normalized = NormalizeForMatch(sentence),
                    Score = themeTerms.Count(term => NormalizeForMatch(sentence).Contains(NormalizeForMatch(term)))
                })
                .Where(item => item.Score > 0)
                .Where(item => !used.Contains(item.Normalized))
                .Where(item => !item.Normalized.Contains("table des matieres"))
                .OrderByDescending(item => item.Score)
                .ThenBy(item => item.Sentence.Length > 340 ? 1 : 0)
                .ThenBy(item => item.Index)
                .Select(item => item.Sentence)
                .FirstOrDefault();
        }

        private static string TranslateAcademicSentenceToEnglish(string sentence)
        {
            var normalized = NormalizeForMatch(sentence);

            if (normalized.Contains("l annee academique comporte deux semestres"))
                return "The academic year has two university semesters of 14 weeks each, with 30 ECTS credits per semester, plus an optional 8-week summer trimester depending on the institution.";

            if (normalized.Contains("temps plein") && normalized.Contains("24 credits"))
                return "A regular student follows a degree program; a full-time student is registered for at least 24 ECTS credits in the current semester, while part-time status starts at 15 ECTS credits.";

            if (normalized.Contains("echoue") && normalized.Contains("deux inscriptions"))
                return "If a student fails a course unit after two registrations, any new registration requires prior jury approval after a meeting between the student and the institution head.";

            if (normalized.Contains("absent") && normalized.Contains("element d evaluation"))
                return "If a student misses an assessment other than the final exam, they must justify the absence in writing with a valid reason within three working days.";

            if (normalized.Contains("un diplome est remis") || (normalized.Contains("diplome") && normalized.Contains("satisfait les exigences")))
                return "A diploma is awarded only after the student validates the required program credits, the USJ general education requirements, and the required course units for the program.";

            if (normalized.Contains("credits auxquels est inscrit") && normalized.Contains("valides"))
                return "Registered credits are validated after the student's learning outcomes are assessed.";

            return sentence;
        }

        private static int ScoreSummarySentence(string sentence, HashSet<string> terms)
        {
            var normalized = NormalizeForMatch(sentence);
            if (normalized.Length < 35)
                return 0;

            if (normalized.Contains("table des matieres") ||
                normalized.Contains("page ") ||
                normalized.Contains("universite saint joseph") ||
                normalized.Contains("version mise") ||
                normalized.Contains("reglement interieur"))
            {
                return 0;
            }

            var score = 0;
            foreach (var term in terms)
            {
                if (normalized.Contains(term))
                    score++;
            }

            if (QuestionContainsAny(normalized, "semester", "semestre", "credit", "credits", "ects", "inscription", "examens", "evaluation"))
                score += 2;

            if (sentence.Length is >= 70 and <= 320)
                score += 1;

            return score;
        }

        private static string PolishSummarySentence(string sentence)
        {
            var value = PolishSnippet(sentence);
            value = Regex.Replace(value, @"^\s*[a-z]\.\s+", "", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            return Preview(value, 340).Trim('-', ';', ',', ' ');
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

            var semesterTimeline = BuildSemesterTimelineAnswer(entries, questionText, citations[0].chunk.Document?.Title ?? "Calendar");
            if (!string.IsNullOrWhiteSpace(semesterTimeline))
                return semesterTimeline;

            var semesterBoundary = BuildSemesterBoundaryAnswer(entries, questionText, citations[0].chunk.Document?.Title ?? "Calendar");
            if (!string.IsNullOrWhiteSpace(semesterBoundary))
                return semesterBoundary;

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

                return $"Here is what the uploaded calendar says in \"{titleFromEntries}\":\n\n{string.Join("\n", lines)}{note}";
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

        private static string? BuildSemesterTimelineAnswer(List<CalendarEntry> entries, string questionText, string documentTitle)
        {
            if (!IsTimelineLike(questionText) || !TryGetSemesterNumber(questionText, out var semesterNumber))
                return null;

            var startEntry = entries
                .Where(entry => EntryLabelContainsAny(entry, $"debut sem {semesterNumber}", $"debut sem. {semesterNumber}"))
                .OrderBy(entry => entry.Date)
                .FirstOrDefault();

            if (startEntry == null)
                return null;

            var nextSemesterStart = entries
                .Where(entry => EntryLabelContainsAny(entry, $"debut sem {semesterNumber + 1}", $"debut sem. {semesterNumber + 1}"))
                .OrderBy(entry => entry.Date)
                .FirstOrDefault();

            var summerStart = entries
                .Where(entry => EntryLabelContainsAny(entry, "debut trim", "trim ete"))
                .OrderBy(entry => entry.Date)
                .FirstOrDefault();

            var endDate = semesterNumber == 1
                ? nextSemesterStart?.Date.AddDays(-1)
                : summerStart?.Date.AddDays(-1);

            var timeline = entries
                .Where(entry => entry.Date >= startEntry.Date && (!endDate.HasValue || entry.Date <= endDate.Value))
                .Where(IsSemesterTimelineEntry)
                .OrderBy(entry => entry.Date)
                .ThenBy(entry => entry.Label)
                .ToList();

            if (!timeline.Any(entry => entry.Date == startEntry.Date && EntryLabelContainsAny(entry, "debut sem")))
                timeline.Insert(0, startEntry);

            var lines = FormatCalendarEntries(timeline)
                .Select(line => $"- {line}")
                .ToList();

            if (nextSemesterStart != null && semesterNumber == 1)
                lines.Add($"- {FormatDateRange(nextSemesterStart.Date, nextSemesterStart.Date)}: Semester 2 starts");
            else if (summerStart != null && semesterNumber == 2)
                lines.Add($"- {FormatDateRange(summerStart.Date, summerStart.Date)}: Summer trimester starts");

            return $"Here is the Semester {semesterNumber} timeline from \"{documentTitle}\":\n\n{string.Join("\n", lines)}";
        }

        private static string? BuildSemesterBoundaryAnswer(List<CalendarEntry> entries, string questionText, string documentTitle)
        {
            if (!IsSemesterBoundaryQuestion(questionText))
                return null;

            var normalizedQuestion = NormalizeForMatch(questionText);
            var wantsEnd = IsEndLike(normalizedQuestion);
            var wantsStart = IsStartLike(normalizedQuestion);

            if (!TryGetSemesterNumber(questionText, out var semesterNumber))
            {
                if (!QuestionContainsAny(normalizedQuestion, "semester", "semesters", "semestre", "semestres", "sem"))
                    return null;

                var lines = new List<string>();
                foreach (var number in new[] { 1, 2 })
                {
                    var semesterStart = FindSemesterStart(entries, number);
                    var followingStart = number == 1 ? FindSemesterStart(entries, 2) : FindSummerStart(entries);

                    if (wantsEnd && semesterStart != null && followingStart != null)
                        lines.Add($"Semester {number} ends on {FormatDateRange(followingStart.Date.AddDays(-1), followingStart.Date.AddDays(-1))}.");
                    else if (wantsStart && semesterStart != null)
                        lines.Add($"Semester {number} starts on {FormatDateRange(semesterStart.Date, semesterStart.Date)}.");
                }

                if (lines.Count == 0)
                    return null;

                var label = wantsEnd ? "end dates" : "start dates";
                return $"Here are the semester {label} from \"{documentTitle}\":\n\n- {string.Join("\n- ", lines)}";
            }

            var startEntry = FindSemesterStart(entries, semesterNumber);
            if (startEntry == null)
                return null;

            var nextStart = semesterNumber == 1
                ? FindSemesterStart(entries, 2)
                : FindSummerStart(entries);

            if (wantsEnd)
            {
                if (wantsStart && nextStart != null)
                {
                    var endDateBoth = nextStart.Date.AddDays(-1);
                    return $"Semester {semesterNumber} starts on {FormatDateRange(startEntry.Date, startEntry.Date)} and ends on {FormatDateRange(endDateBoth, endDateBoth)} in \"{documentTitle}\". I infer the end date from the next academic period starting on {FormatDateRange(nextStart.Date, nextStart.Date)}.";
                }

                if (nextStart == null)
                    return $"Semester {semesterNumber} starts on {FormatDateRange(startEntry.Date, startEntry.Date)} in \"{documentTitle}\", but I could not infer the semester end from the uploaded calendar.";

                var endDate = nextStart.Date.AddDays(-1);
                var nextLabel = semesterNumber == 1 ? "Semester 2 starts" : "the summer trimester starts";
                var examEntries = entries
                    .Where(entry => entry.Date >= startEntry.Date && entry.Date <= endDate && EntryLabelContainsAny(entry, "examen", "examens", "langues"))
                    .OrderBy(entry => entry.Date)
                    .ToList();
                var examLines = FormatCalendarEntries(examEntries)
                    .Take(6)
                    .Select(line => $"- {line}")
                    .ToList();

                var examText = examLines.Count > 0
                    ? $"\n\nMain exam/language dates inside this semester:\n{string.Join("\n", examLines)}"
                    : "";

                return $"Semester {semesterNumber} ends on {FormatDateRange(endDate, endDate)} in \"{documentTitle}\". I infer that because {nextLabel} on {FormatDateRange(nextStart.Date, nextStart.Date)}.{examText}";
            }

            if (wantsStart)
            {
                return $"Semester {semesterNumber} starts on {FormatDateRange(startEntry.Date, startEntry.Date)} in \"{documentTitle}\": {startEntry.Label}.";
            }

            return null;
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
            var wantsTimeline = IsTimelineLike(questionText);
            var wantsEnd = IsEndLike(normalizedQuestion);

            if (TryGetSemesterNumber(questionText, out var requestedSemester) &&
                TryGetSemesterDateRange(entries, requestedSemester, out var semesterStart, out var semesterEnd))
            {
                query = query.Where(entry => entry.Date >= semesterStart && entry.Date <= semesterEnd);
            }

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
                if (wantsEnd && TryGetSemesterNumber(questionText, out var endSemesterNumber))
                {
                    var startEntry = FindSemesterStart(entries, endSemesterNumber);
                    var nextEntry = endSemesterNumber == 1 ? FindSemesterStart(entries, 2) : FindSummerStart(entries);

                    if (startEntry != null && nextEntry != null)
                    {
                        var endDate = nextEntry.Date.AddDays(-1);
                        query = query.Where(entry => entry.Date >= startEntry.Date && entry.Date <= endDate && IsSemesterTimelineEntry(entry));
                    }
                    else
                    {
                        query = query.Where(entry => EntryLabelContainsAny(entry, "debut sem"));
                    }
                }
                else if (wantsTimeline && TryGetSemesterNumber(questionText, out var semesterNumber))
                {
                    var startEntry = entries
                        .Where(entry => EntryLabelContainsAny(entry, $"debut sem {semesterNumber}", $"debut sem. {semesterNumber}"))
                        .OrderBy(entry => entry.Date)
                        .FirstOrDefault();

                    var nextEntry = entries
                        .Where(entry => EntryLabelContainsAny(entry, $"debut sem {semesterNumber + 1}", $"debut sem. {semesterNumber + 1}"))
                        .OrderBy(entry => entry.Date)
                        .FirstOrDefault();

                    if (startEntry != null)
                    {
                        var endDate = nextEntry?.Date.AddDays(-1);
                        query = query.Where(entry => entry.Date >= startEntry.Date && (!endDate.HasValue || entry.Date <= endDate.Value) && IsSemesterTimelineEntry(entry));
                    }
                    else
                    {
                        query = query.Where(entry => EntryLabelContainsAny(entry, "debut sem"));
                    }
                }
                else
                {
                    query = query.Where(entry => EntryLabelContainsAny(entry, "debut sem"));
                }

                if (!wantsTimeline && (normalizedQuestion.Contains("semester 1") || normalizedQuestion.Contains("semestre 1") || normalizedQuestion.Contains("sem 1")))
                    query = query.Where(entry => NormalizeForMatch(entry.Label).Contains("1"));
                else if (!wantsTimeline && (normalizedQuestion.Contains("semester 2") || normalizedQuestion.Contains("semestre 2") || normalizedQuestion.Contains("sem 2")))
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

        public static string FormatSourcePreview(string text, int maxLength = 420)
        {
            var cleaned = CleanTextForAnswer(text ?? "");
            return Preview(cleaned, maxLength);
        }

        public static string BuildEvidencePreview(string chunkText, string questionText, string answerText, int maxLength = 720)
        {
            var answerLines = (answerText ?? "")
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(line => Regex.IsMatch(line, @"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b\d{1,2}[-–]\d{1,2},\s*\d{4}\b", RegexOptions.IgnoreCase))
                .Select(line => line.Trim('-', ' ', '\t'))
                .Take(6)
                .ToList();

            if (answerLines.Count > 0 && LooksLikeCalendarQuestion(questionText))
                return string.Join("\n", answerLines);

            var answerBullets = (answerText ?? "")
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(line => line.StartsWith("- ", StringComparison.Ordinal))
                .Select(line => line.Trim('-', ' ', '\t'))
                .Where(line => line.Length > 30)
                .Take(4)
                .ToList();

            if (answerBullets.Count > 0 && (IsSummaryLike(questionText) || TryGetFollowUpTransformInstruction(questionText, out _)))
                return string.Join("\n", answerBullets);

            var cleaned = CleanTextForAnswer(chunkText ?? "");
            var highlights = BuildEvidenceHighlights(questionText, answerText ?? "");
            var sentences = Regex
                .Split(cleaned, @"(?<=[.!?])\s+(?=[A-Z0-9À-ÖØ-Þ])", RegexOptions.CultureInvariant)
                .Select(sentence => PolishSnippet(sentence))
                .Where(sentence => sentence.Length > 35)
                .Select((sentence, index) => new
                {
                    Sentence = sentence,
                    Index = index,
                    Score = ScoreEvidenceSentence(sentence, highlights)
                })
                .Where(item => item.Score > 0)
                .OrderByDescending(item => item.Score)
                .ThenBy(item => item.Index)
                .Take(3)
                .OrderBy(item => item.Index)
                .Select(item => item.Sentence)
                .ToList();

            var evidence = sentences.Count > 0
                ? string.Join(" ", sentences)
                : ExtractBestSnippet(cleaned, highlights, maxLength);

            return Preview(evidence, maxLength);
        }

        public static List<string> BuildEvidenceHighlights(string questionText, string answerText)
        {
            var terms = new List<string>();

            foreach (Match match in Regex.Matches(answerText ?? "", @"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:[-–]\d{1,2})?,\s*\d{4}\b", RegexOptions.IgnoreCase))
                terms.Add(match.Value);

            var highlightSource = TryGetFollowUpTransformInstruction(questionText, out _)
                ? answerText
                : $"{questionText} {answerText}";

            foreach (var token in Tokenize(highlightSource ?? ""))
            {
                var normalized = NormalizeForMatch(token);
                if (normalized.Length < 4 || GenericAnswerTerms.Contains(normalized))
                    continue;

                if (!terms.Any(term => NormalizeForMatch(term) == normalized))
                    terms.Add(token);
            }

            return terms
                .Where(term => !string.IsNullOrWhiteSpace(term))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Take(14)
                .ToList();
        }

        private static int ScoreEvidenceSentence(string sentence, List<string> highlights)
        {
            var normalized = NormalizeForMatch(sentence);
            var score = 0;

            foreach (var highlight in highlights)
            {
                var term = NormalizeForMatch(highlight);
                if (term.Length > 2 && normalized.Contains(term))
                    score++;
            }

            if (sentence.Length is >= 60 and <= 360)
                score++;

            return score;
        }

        private static string HumanizeAnswerText(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return "I could not find enough reliable information in the uploaded university documents.";

            var cleaned = text.Trim();
            cleaned = Regex.Replace(cleaned, @"^\s*answer\s*:\s*", "", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant).Trim();
            cleaned = cleaned.Trim('"', '\'', ' ', '\n', '\r', '\t');
            cleaned = Regex.Replace(cleaned, @"\n{3,}", "\n\n");

            if (string.IsNullOrWhiteSpace(cleaned))
                return "I could not find enough reliable information in the uploaded university documents.";

            return cleaned;
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

        private static bool IsSemesterTimelineEntry(CalendarEntry entry)
        {
            return EntryLabelContainsAny(entry, "debut sem", "examen", "langues", "rdd") || IsHolidayEntry(entry);
        }

        private static CalendarEntry? FindSemesterStart(List<CalendarEntry> entries, int semesterNumber)
        {
            return entries
                .Where(entry => EntryLabelContainsAny(entry, $"debut sem {semesterNumber}", $"debut sem. {semesterNumber}"))
                .OrderBy(entry => entry.Date)
                .FirstOrDefault();
        }

        private static CalendarEntry? FindSummerStart(List<CalendarEntry> entries)
        {
            return entries
                .Where(entry => EntryLabelContainsAny(entry, "debut trim", "trim ete"))
                .OrderBy(entry => entry.Date)
                .FirstOrDefault();
        }

        private static bool TryGetSemesterDateRange(List<CalendarEntry> entries, int semesterNumber, out DateTime start, out DateTime end)
        {
            start = default;
            end = default;

            var startEntry = FindSemesterStart(entries, semesterNumber);
            if (startEntry == null)
                return false;

            var nextStart = semesterNumber == 1
                ? FindSemesterStart(entries, 2)
                : FindSummerStart(entries);

            start = startEntry.Date;
            end = nextStart?.Date.AddDays(-1) ?? DateTime.MaxValue;
            return true;
        }

        private static bool IsTimelineLike(string text)
        {
            var normalized = NormalizeForMatch(text);
            return normalized.Contains("timeline") ||
                   normalized.Contains("time line") ||
                   normalized.Contains("chronology") ||
                   normalized.Contains("calendar for") ||
                   normalized.Contains("all dates") ||
                   normalized.Contains("important dates") ||
                   normalized.Contains("correct timeline");
        }

        private static bool IsSemesterBoundaryQuestion(string text)
        {
            var normalized = NormalizeForMatch(text);
            return IsStartLike(normalized) || IsEndLike(normalized);
        }

        private static bool IsStartLike(string normalizedText)
        {
            return QuestionContainsAny(normalizedText, "start", "starts", "begin", "begins", "beginning", "begining", "debut");
        }

        private static bool IsEndLike(string normalizedText)
        {
            return QuestionContainsAny(normalizedText, "end", "ends", "ending", "finish", "finishes", "finished", "close", "closes", "fin");
        }

        private static bool TryGetSemesterNumber(string text, out int semesterNumber)
        {
            var normalized = NormalizeForMatch(text);
            if (Regex.IsMatch(normalized, @"\b(semester|semestre|sem)\s*1\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
            {
                semesterNumber = 1;
                return true;
            }

            if (Regex.IsMatch(normalized, @"\b(semester|semestre|sem)\s*2\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
            {
                semesterNumber = 2;
                return true;
            }

            semesterNumber = 0;
            return false;
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
                @"\b(Vend|Lun|Mar|Mer|Jeu|Ven|Sam|Dim)\s*(\d{1,2})(?=\D|$)",
                "$1 $2 ",
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
            var cleaned = CleanTextForAnswer(text);
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

            var lower = RemoveAccents(cleaned).ToLowerInvariant();
            var indexes = terms
                .Select(term => lower.IndexOf(NormalizeForMatch(term), StringComparison.OrdinalIgnoreCase))
                .Where(index => index >= 0)
                .ToList();

            if (indexes.Count == 0)
                return Preview(cleaned, maxLength);

            var first = indexes.Min();
            var start = FindNaturalSnippetStart(cleaned, first, 180);
            var length = Math.Min(maxLength, cleaned.Length - start);
            var snippet = cleaned.Substring(start, length).Trim();

            if (start + length < cleaned.Length)
                snippet += "...";

            return PolishSnippet(snippet);
        }

        private static int FindNaturalSnippetStart(string text, int termIndex, int maxLead)
        {
            var minStart = Math.Max(0, termIndex - maxLead);
            var boundaryStart = minStart;

            foreach (var boundary in new[] { ". ", "? ", "! ", "\n" })
            {
                var index = text.LastIndexOf(boundary, Math.Max(0, termIndex - 1), termIndex - minStart, StringComparison.Ordinal);
                if (index >= minStart)
                    boundaryStart = Math.Max(boundaryStart, index + boundary.Length);
            }

            return boundaryStart;
        }

        private static int FindSectionStart(string text, List<string> terms)
        {
            var headingMap = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["objective"] = new[] { "OBJECTIVE" },
                ["aim"] = new[] { "AIM" },
                ["scope"] = new[] { "SCOPE" },
                ["objet"] = new[] { "OBJET" },
                ["responsibility"] = new[] { "RESPONSIBILITIES", "RESPONSIBILITY" },
                ["responsibilities"] = new[] { "RESPONSIBILITIES", "RESPONSIBILITY" },
                ["rule"] = new[] { "CCTV RULES", "RULES AND REGULATIONS", "RULES" },
                ["rules"] = new[] { "CCTV RULES", "RULES AND REGULATIONS", "RULES" },
                ["implementation"] = new[] { "CCTV IMPLEMENTATION", "IMPLEMENTATION" },
                ["access"] = new[] { "ACCESS REQUESTS", "ACCESS FORM" },
                ["requests"] = new[] { "ACCESS REQUESTS", "ACCESS FORM" },
                ["retention"] = new[] { "RETENTION" },
                ["ethique"] = new[] { "ETHIQUE UNIVERSITAIRE", "ÉTHIQUE UNIVERSITAIRE", "OBLIGATIONS DES ETUDIANTS", "OBLIGATIONS DES ÉTUDIANTS" },
                ["obligation"] = new[] { "OBLIGATIONS DES ETUDIANTS", "OBLIGATIONS DES ÉTUDIANTS" },
                ["obligations"] = new[] { "OBLIGATIONS DES ETUDIANTS", "OBLIGATIONS DES ÉTUDIANTS" },
                ["amicale"] = new[] { "LES AMICALES", "OBJET" },
                ["assembly"] = new[] { "ASSEMBLEES GENERALES", "ASSEMBLÉES GÉNÉRALES", "ASSEMBLEE GENERALE", "ASSEMBLÉE GÉNÉRALE" },
                ["assemblee"] = new[] { "ASSEMBLEES GENERALES", "ASSEMBLÉES GÉNÉRALES", "ASSEMBLEE GENERALE", "ASSEMBLÉE GÉNÉRALE" },
                ["quorum"] = new[] { "QUORUM" },
                ["bureau"] = new[] { "BUREAU DE L'AMICALE", "BUREAU" },
                ["committee"] = new[] { "BUREAU DE L'AMICALE", "BUREAU" },
                ["composition"] = new[] { "COMPOSITION" },
                ["mandate"] = new[] { "DUREE DU MANDAT", "DURÉE DU MANDAT", "MANDAT" },
                ["mandat"] = new[] { "DUREE DU MANDAT", "DURÉE DU MANDAT", "MANDAT" },
                ["election"] = new[] { "ELECTIONS DU BUREAU", "ÉLECTIONS DU BUREAU", "ELECTION", "ÉLECTION" },
                ["elections"] = new[] { "ELECTIONS DU BUREAU", "ÉLECTIONS DU BUREAU", "ELECTION", "ÉLECTION" },
                ["vote"] = new[] { "ELECTIONS DU BUREAU", "ÉLECTIONS DU BUREAU", "SCRUTIN", "VOTE" },
                ["bureau electoral"] = new[] { "BUREAU ELECTORAL", "BUREAU ÉLECTORAL" },
                ["listes des electeurs"] = new[] { "LISTES DES ELECTEURS", "LISTES DES ÉLECTEURS" },
                ["access requests"] = new[] { "ACCESS REQUESTS" },
                ["quorum requis"] = new[] { "QUORUM" },
                ["candidate"] = new[] { "CANDIDATURE", "CANDIDAT" },
                ["candidature"] = new[] { "CANDIDATURE", "CANDIDAT" },
                ["finance"] = new[] { "DISPOSITIONS FINANCIERES", "DISPOSITIONS FINANCIÈRES", "TRESORIER", "TRÉSORIER" },
                ["financial"] = new[] { "DISPOSITIONS FINANCIERES", "DISPOSITIONS FINANCIÈRES", "TRESORIER", "TRÉSORIER" },
                ["president"] = new[] { "PRESIDENT", "PRÉSIDENT" },
                ["secretary"] = new[] { "SECRETAIRE", "SECRÉTAIRE" },
                ["secretaire"] = new[] { "SECRETAIRE", "SECRÉTAIRE" },
                ["treasurer"] = new[] { "TRESORIER", "TRÉSORIER" },
                ["tresorier"] = new[] { "TRESORIER", "TRÉSORIER" }
            };

            foreach (var term in terms.Select(NormalizeForMatch))
            {
                if (!headingMap.TryGetValue(term, out var headings))
                    continue;

                foreach (var heading in headings)
                {
                    var escapedHeading = Regex.Escape(heading).Replace("\\ ", "\\s+");
                    var patterns = new[]
                    {
                        $@"(?:^|\s)(?:article|ARTICLE)\s+[A-ZIVX0-9]+\s*(?:[:\-–]\s*){escapedHeading}\b",
                        $@"(?:^|\s)\d+(?:\.\d+)?\s*\.?\s*{escapedHeading}\b",
                        $@"(?:^|\s)(?:titre|TITRE)\s+[IVX]+\s*[–\-]\s*[^\n\r\.]{{0,90}}{escapedHeading}\b"
                    };

                    var matches = patterns
                        .SelectMany(pattern => Regex.Matches(text, pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
                        .OrderBy(match => match.Index)
                        .ToList();

                    if (matches.Count > 0)
                        return matches[^1].Index;
                }
            }

            return -1;
        }

        private static string CleanTextForAnswer(string text)
        {
            var value = NormalizeCalendarText(text ?? "");
            value = value.Replace("des amicales d'étudiantsdes institutions", "des amicales d'étudiants des institutions", StringComparison.OrdinalIgnoreCase);
            value = value.Replace("TABLE DES MATIÈRESP", "TABLE DES MATIÈRES P", StringComparison.OrdinalIgnoreCase);
            value = value.Replace("TABLE DES MATIERE P", "TABLE DES MATIÈRES P", StringComparison.OrdinalIgnoreCase);
            value = value.Replace("BEYROUTHCALENDRIER", "BEYROUTH CALENDRIER", StringComparison.OrdinalIgnoreCase);
            value = value.Replace("2025-2026Ce", "2025-2026 Ce", StringComparison.OrdinalIgnoreCase);
            value = Regex.Replace(value, @"(?<=\p{L})(?=Article\s+\d)", " ", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=\p{L})(?=Titre\s+[IVX]+)", " ", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=\p{L})(?=Version\s+mise)", " ", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=\d)(?=\p{Lu})", " ", RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=\p{L})(?=\d+\s*Version)", " ", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"(?<=[.;:])(?=\S)", " ", RegexOptions.CultureInvariant);
            value = Regex.Replace(value, @"\s*•\s*", " - ", RegexOptions.CultureInvariant);
            return PolishSnippet(value);
        }

        private static string PolishSnippet(string text)
        {
            var value = text ?? "";
            value = value.Replace("BEYROUTHCALENDRIER", "BEYROUTH CALENDRIER", StringComparison.OrdinalIgnoreCase);
            value = value.Replace("2025-2026Ce", "2025-2026 Ce", StringComparison.OrdinalIgnoreCase);
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
            var cleaned = CleanTextForAnswer(chunk.Text);
            var text = NormalizeForMatch(cleaned);
            var title = NormalizeForMatch(chunk.Document?.Title);
            var metadata = NormalizeForMatch($"{chunk.Document?.Department} {chunk.Document?.Course} {chunk.Document?.Category} {chunk.Document?.Tags}");
            var question = NormalizeForMatch(questionText);

            float score = 0;

            if (question.Length > 8 && text.Contains(question))
                score += 12;

            foreach (var term in terms)
            {
                var normalizedTerm = NormalizeForMatch(term);
                if (normalizedTerm.Length <= 2)
                    continue;

                var count = CountOccurrences(text, normalizedTerm);
                if (count > 0)
                    score += Math.Min(count, 6);

                if (title.Contains(normalizedTerm))
                    score += 4;

                if (metadata.Contains(normalizedTerm))
                    score += 2.5f;

                if (LooksLikeHeadingMatch(cleaned, normalizedTerm))
                    score += 5;
            }

            if (text.Contains("table des matieres") || text.Contains("table of content"))
                score -= 0.75f;

            return score;
        }

        private static int CountOccurrences(string text, string term)
        {
            var count = 0;
            var index = text.IndexOf(term, StringComparison.OrdinalIgnoreCase);

            while (index >= 0)
            {
                count++;
                index = text.IndexOf(term, index + term.Length, StringComparison.OrdinalIgnoreCase);
            }

            return count;
        }

        private static bool LooksLikeHeadingMatch(string cleanedText, string normalizedTerm)
        {
            var pattern = $@"(?:^|\s)(?:article\s+\d+\s*[:\-–]\s*)?(?:\d+(?:\.\d+)?\.?\s*)?{Regex.Escape(normalizedTerm)}(?:\s|[:\-–])";
            return Regex.IsMatch(NormalizeForMatch(cleanedText), pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
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

                    case "end":
                    case "ends":
                    case "ending":
                    case "finish":
                    case "finishes":
                    case "finished":
                    case "close":
                    case "closes":
                    case "fin":
                        AddTerms(expanded, "end", "ends", "ending", "finish", "finished", "close", "closes", "fin", "debut");
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

                    case "policy":
                    case "policies":
                    case "politique":
                    case "procedure":
                    case "procedures":
                        AddTerms(expanded, "policy", "policies", "politique", "procedure", "procedures", "regles", "rules");
                        break;

                    case "objective":
                    case "objectives":
                    case "purpose":
                    case "objet":
                        AddTerms(expanded, "objective", "objectives", "purpose", "objet");
                        break;

                    case "aim":
                    case "goal":
                    case "goals":
                        AddTerms(expanded, "aim", "goal", "goals");
                        break;

                    case "scope":
                    case "applies":
                    case "apply":
                        AddTerms(expanded, "scope", "applies", "apply", "applicable");
                        break;

                    case "responsibility":
                    case "responsibilities":
                    case "responsible":
                    case "role":
                    case "roles":
                        AddTerms(expanded, "responsibility", "responsibilities", "responsible", "role", "roles", "charge");
                        break;

                    case "privacy":
                    case "confidentiality":
                    case "confidential":
                    case "security":
                    case "secure":
                        AddTerms(expanded, "privacy", "confidentiality", "confidential", "security", "secure", "protection", "securite");
                        break;

                    case "recording":
                    case "recordings":
                    case "camera":
                    case "cameras":
                    case "monitoring":
                    case "cctv":
                        AddTerms(expanded, "recording", "recordings", "camera", "cameras", "monitoring", "cctv", "images", "surveillance");
                        break;

                    case "storage":
                    case "retain":
                    case "retained":
                    case "retention":
                    case "delete":
                    case "deleted":
                        AddTerms(expanded, "storage", "retain", "retained", "retention", "delete", "deleted", "conserver", "archives");
                        break;

                    case "request":
                    case "requests":
                    case "access":
                    case "disclose":
                    case "disclosure":
                    case "approval":
                    case "approved":
                        AddTerms(expanded, "request", "requests", "access", "disclose", "disclosure", "approval", "approved", "demande", "dpo");
                        break;

                    case "election":
                    case "elections":
                    case "electoral":
                    case "vote":
                    case "voting":
                    case "scrutin":
                        AddTerms(expanded, "election", "elections", "electoral", "vote", "voting", "scrutin", "suffrage", "bureau electoral");
                        break;

                    case "candidate":
                    case "candidates":
                    case "candidature":
                    case "candidatures":
                    case "candidat":
                    case "candidats":
                        AddTerms(expanded, "candidate", "candidates", "candidature", "candidatures", "candidat", "candidats");
                        break;

                    case "association":
                    case "amicale":
                    case "club":
                    case "student":
                    case "students":
                    case "etudiant":
                    case "etudiants":
                        AddTerms(expanded, "association", "amicale", "club", "student", "students", "etudiant", "etudiants");
                        break;

                    case "member":
                    case "members":
                    case "membership":
                    case "membre":
                    case "membres":
                        AddTerms(expanded, "member", "members", "membership", "membre", "membres");
                        break;

                    case "assembly":
                    case "assemblies":
                    case "assemblee":
                    case "assemblees":
                    case "meeting":
                    case "meetings":
                    case "reunion":
                    case "reunions":
                    case "quorum":
                        AddTerms(expanded, "assembly", "assemblies", "assemblee", "assemblees", "meeting", "meetings", "reunion", "reunions", "quorum");
                        break;

                    case "bureau":
                    case "committee":
                    case "committees":
                    case "office":
                    case "board":
                        AddTerms(expanded, "bureau", "committee", "committees", "office", "board");
                        break;

                    case "president":
                    case "vice":
                    case "secretary":
                    case "secretaire":
                    case "treasurer":
                    case "tresorier":
                        AddTerms(expanded, "president", "vice", "secretary", "secretaire", "treasurer", "tresorier");
                        break;

                    case "mandate":
                    case "mandat":
                    case "duration":
                    case "duree":
                    case "term":
                        AddTerms(expanded, "mandate", "mandat", "duration", "duree", "term");
                        break;

                    case "resignation":
                    case "resign":
                    case "demission":
                    case "demissionnaire":
                        AddTerms(expanded, "resignation", "resign", "demission", "demissionnaire");
                        break;

                    case "finance":
                    case "financial":
                    case "budget":
                    case "budgets":
                    case "money":
                    case "financieres":
                    case "financier":
                        AddTerms(expanded, "finance", "financial", "budget", "budgets", "money", "financieres", "financier", "tresorier");
                        break;

                    case "ethics":
                    case "ethical":
                    case "ethique":
                    case "obligation":
                    case "obligations":
                        AddTerms(expanded, "ethics", "ethical", "ethique", "obligation", "obligations", "reserve", "tenue");
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

        private sealed record PreviousTurn(
            string QuestionText,
            string AnswerText,
            List<(DocumentChunk chunk, float score)> Citations);

        private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
        {
            "the", "and", "for", "are", "you", "your", "with", "that", "this", "what", "when",
            "where", "who", "why", "how", "can", "could", "should", "would", "about", "into",
            "from", "does", "have", "has", "had", "will", "shall", "may", "might", "must",
            "please", "tell", "explain", "give", "show", "document", "documents", "is", "was",
            "were", "am", "a", "an", "of", "to", "in", "on", "at", "by", "me", "my", "our",
            "their", "it", "its", "as", "or"
        };

        private static readonly HashSet<string> GenericAnswerTerms = new(StringComparer.OrdinalIgnoreCase)
        {
            "document", "documents", "uploaded", "university", "universite", "saint", "joseph",
            "beirut", "beyrouth", "policy", "policies", "statuts", "status", "calendar",
            "calendrier", "handled", "work", "works", "correct", "give", "tell"
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

        private class RerankRequest
        {
            [JsonPropertyName("question")]
            public string Question { get; set; } = "";

            [JsonPropertyName("chunks")]
            public List<RerankChunkItem> Chunks { get; set; } = new();
        }

        private class RerankChunkItem
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

        private class RerankResponse
        {
            [JsonPropertyName("results")]
            public List<RerankResult> Results { get; set; } = new();
        }

        private class RerankResult
        {
            [JsonPropertyName("chunk_id")]
            public int ChunkId { get; set; }

            [JsonPropertyName("score")]
            public float Score { get; set; }
        }

        private class TransformRequest
        {
            [JsonPropertyName("instruction")]
            public string Instruction { get; set; } = "";

            [JsonPropertyName("text")]
            public string Text { get; set; } = "";
        }

        private class TransformResponse
        {
            [JsonPropertyName("answer")]
            public string Answer { get; set; } = "";
        }
    }
}
