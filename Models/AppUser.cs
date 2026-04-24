namespace Fyp.Models;

public class AppUser
{
    public int Id { get; set; }
    public string FirstName { get; set; } = "";
    public string LastName { get; set; } = "";
    public string StudentNumber { get; set; } = "";
    public string Email { get; set; } = "";
    public string Department { get; set; } = "";
    public bool TermsAccepted { get; set; }
    public string PasswordHash { get; set; } = "";
    public string Role { get; set; } = "Student";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public List<Question> Questions { get; set; } = new();
}
