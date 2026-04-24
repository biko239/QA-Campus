namespace Fyp.Models
{
    public class AdminUser
    {
        public int Id { get; set; }

        public string Username { get; set; } = "";
        public string PasswordHash { get; set; } = "";
        public string AccessCodeHash { get; set; } = "";

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}