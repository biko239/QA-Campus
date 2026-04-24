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
                answer = result.answer.Text,
                citations = result.citations.Select(c => new
                {
                    documentTitle = c.chunk.Document?.Title ?? "Unknown Document",
                    chunkId = c.chunk.Id,
                    score = c.score,
                    preview = c.chunk.Text.Length > 300 ? c.chunk.Text[..300] + "..." : c.chunk.Text
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
