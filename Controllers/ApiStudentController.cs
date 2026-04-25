using Fyp.Data;
using Fyp.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Controllers;

[ApiController]
[Route("api/student")]
public class ApiStudentController : ControllerBase
{
    private readonly AppDbContext _db;
    private readonly RagService _rag;

    public ApiStudentController(AppDbContext db, RagService rag)
    {
        _db = db;
        _rag = rag;
    }

    [HttpGet("documents")]
    public async Task<IActionResult> PublicDocuments()
    {
        if (HttpContext.Session.GetString("Role") != "Student")
            return Unauthorized(new { message = "Student login required." });

        var docs = await _db.Documents
            .Where(d => d.IsPublic && d.Status == "READY")
            .OrderByDescending(d => d.UploadedAt)
            .Select(d => new
            {
                d.Id,
                d.Title,
                d.Department,
                d.Category
            })
            .ToListAsync();

        return Ok(docs);
    }

    [HttpGet("chat/history")]
    public async Task<IActionResult> ChatHistory()
    {
        var role = HttpContext.Session.GetString("Role");
        var userId = HttpContext.Session.GetInt32("UserId");

        if (role != "Student" || userId == null)
            return Unauthorized(new { message = "Student login required." });

        var rows = await _db.Questions
            .AsNoTracking()
            .Include(q => q.Answer)
                .ThenInclude(a => a!.ChunkUsages)
                    .ThenInclude(cu => cu.Chunk)
                        .ThenInclude(c => c!.Document)
            .Where(q => q.UserId == userId.Value && q.Answer != null)
            .OrderByDescending(q => q.AskedAt)
            .Take(50)
            .ToListAsync();

        var history = rows
            .OrderBy(q => q.AskedAt)
            .Select(q => new
            {
                id = q.Id,
                question = q.Text,
                askedAt = q.AskedAt,
                answer = q.Answer!.Text,
                citations = q.Answer.ChunkUsages.Select(cu => new
                {
                    documentTitle = cu.Chunk?.Document?.Title ?? "Unknown Document",
                    chunkId = cu.ChunkId,
                    score = cu.Score,
                    preview = RagService.FormatSourcePreview(cu.Chunk?.Text ?? ""),
                    evidence = RagService.BuildEvidencePreview(cu.Chunk?.Text ?? "", q.Text, q.Answer.Text),
                    highlights = RagService.BuildEvidenceHighlights(q.Text, q.Answer.Text)
                })
            });

        return Ok(history);
    }

    [HttpPost("ask")]
    public async Task<IActionResult> Ask(AskRequest request)
    {
        var role = HttpContext.Session.GetString("Role");
        var userId = HttpContext.Session.GetInt32("UserId");

        if (role != "Student" || userId == null)
            return Unauthorized(new { message = "Student login required." });

        if (string.IsNullOrWhiteSpace(request.Question))
            return BadRequest(new { message = "Please enter a question." });

        try
        {
            var result = await _rag.AskAsync(userId.Value, request.Question);

            return Ok(new
            {
                id = result.answer.QuestionId,
                answerId = result.answer.Id,
                question = request.Question.Trim(),
                askedAt = DateTime.UtcNow,
                answer = result.answer.Text,
                citations = result.citations.Select(c => new
                {
                    documentTitle = c.chunk.Document?.Title ?? "Unknown Document",
                    chunkId = c.chunk.Id,
                    score = c.score,
                    preview = RagService.FormatSourcePreview(c.chunk.Text),
                    evidence = RagService.BuildEvidencePreview(c.chunk.Text, request.Question, result.answer.Text),
                    highlights = RagService.BuildEvidenceHighlights(request.Question, result.answer.Text)
                })
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new
            {
                message = "The chatbot could not answer right now. Please make sure documents are uploaded and the AI service is running.",
                detail = ex.Message
            });
        }
    }
}

public record AskRequest(string Question);
