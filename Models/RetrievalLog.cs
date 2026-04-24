namespace Fyp.Models;

public class RetrievalLog
{
    public int Id { get; set; }
    public int QuestionId { get; set; }
    public int UserId { get; set; }
    public string SessionId { get; set; } = "";
    public string QueryText { get; set; } = "";
    public int TopK { get; set; }
    public float SimilarityThreshold { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}
