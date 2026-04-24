namespace Fyp.Models;

public class ProcessingLog
{
    public int Id { get; set; }
    public int DocumentId { get; set; }
    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public long DurationMs { get; set; }
    public string Status { get; set; } = "";
    public string Notes { get; set; } = "";
}
