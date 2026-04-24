namespace Fyp.Models;

public class Feedback
{
    public int Id { get; set; }
    public int AnswerId { get; set; }
    public Answer? Answer { get; set; }
    public int UserId { get; set; }
    public AppUser? User { get; set; }
    public string Value { get; set; } = "helpful";
    public string? Comment { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
