namespace Fyp.Models;

public class Answer
{
    public int Id { get; set; }
    public int QuestionId { get; set; }
    public Question? Question { get; set; }
    public string Text { get; set; } = "";
    public float Confidence { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public List<ChunkUsage> ChunkUsages { get; set; } = new();
}
