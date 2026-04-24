namespace Fyp.Models;

public class Question
{
    public int Id { get; set; }
    public int UserId { get; set; }
    public AppUser? User { get; set; }
    public string Text { get; set; } = "";
    public string SessionId { get; set; } = "";
    public DateTime AskedAt { get; set; } = DateTime.UtcNow;
    public Answer? Answer { get; set; }
}
