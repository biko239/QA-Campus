namespace Fyp.Models;

public class DocumentChunk
{
    public int Id { get; set; }
    public int DocumentId { get; set; }
    public Document? Document { get; set; }
    public int ChunkIndex { get; set; }
    public string Text { get; set; } = "";
    public int? PageNumber { get; set; }
    public int? StartOffset { get; set; }
    public int? EndOffset { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public List<ChunkUsage> ChunkUsages { get; set; } = new();
}
