using Fyp.Services;
using Microsoft.AspNetCore.Mvc;

namespace Fyp.Controllers;

[ApiController]
[Route("api/admin")]
public class ApiAdminController : ControllerBase
{
    private readonly AnalyticsService _analytics;
    private readonly DocumentService _docs;
    private readonly RagService _rag;

    public ApiAdminController(AnalyticsService analytics, DocumentService docs, RagService rag)
    {
        _analytics = analytics;
        _docs = docs;
        _rag = rag;
    }

    [HttpGet("dashboard")]
    public async Task<IActionResult> Dashboard()
    {
        if (!IsAdmin())
            return Unauthorized(new { message = "Admin login required." });

        return Ok(await _analytics.GetDashboardStatsAsync());
    }

    [HttpGet("documents")]
    public async Task<IActionResult> Documents()
    {
        if (!IsAdmin())
            return Unauthorized(new { message = "Admin login required." });

        var docs = await _docs.ListAsync();
        return Ok(docs.Select(d => new
        {
            d.Id,
            d.Title,
            d.Department,
            d.Course,
            d.Category,
            d.Description,
            d.Tags,
            d.IsPublic,
            d.FileType,
            d.FileSize,
            d.Status,
            d.ProcessingMethod,
            d.UploadedAt,
            d.ProcessingStartedAt,
            d.ProcessingCompletedAt
        }));
    }

    [HttpGet("documents/{id:int}")]
    public async Task<IActionResult> DocumentDetails(int id)
    {
        if (!IsAdmin())
            return Unauthorized(new { message = "Admin login required." });

        var doc = await _docs.GetAsync(id);
        if (doc == null)
            return NotFound(new { message = "Document not found." });

        return Ok(new
        {
            doc.Id,
            doc.Title,
            doc.Department,
            doc.Course,
            doc.Category,
            doc.Description,
            doc.Tags,
            doc.IsPublic,
            doc.FileType,
            doc.FileSize,
            doc.Status,
            doc.ProcessingMethod,
            doc.UploadedAt,
            doc.ProcessingStartedAt,
            doc.ProcessingCompletedAt,
            chunks = doc.Chunks
                .OrderBy(c => c.ChunkIndex)
                .Select(c => new
                {
                    c.Id,
                    c.ChunkIndex,
                    c.Text,
                    c.PageNumber,
                    c.CreatedAt
                })
        });
    }

    [HttpPost("documents")]
    [RequestSizeLimit(50_000_000)]
    public async Task<IActionResult> UploadDocument(
        IFormFile file,
        [FromForm] string? title,
        [FromForm] string? department,
        [FromForm] string? course,
        [FromForm] string? category,
        [FromForm] string? description,
        [FromForm] string? tags,
        [FromForm] bool isPublic)
    {
        if (!IsAdmin())
            return Unauthorized(new { message = "Admin login required." });

        if (file == null || file.Length == 0)
            return BadRequest(new { message = "Please choose a file." });

        var doc = await _docs.CreateDocumentAsync(
            file,
            title ?? "",
            department ?? "",
            course ?? "",
            category ?? "",
            description ?? "",
            tags ?? "",
            isPublic);

        await _rag.ProcessDocumentAsync(doc.Id);

        return Ok(new { message = "Document uploaded and processed successfully.", documentId = doc.Id });
    }

    private bool IsAdmin() => HttpContext.Session.GetString("Role") == "Admin";
}
