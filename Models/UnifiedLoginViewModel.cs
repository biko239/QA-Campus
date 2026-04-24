namespace Fyp.Models
{
    public class UnifiedLoginViewModel
    {
        public string Identifier { get; set; } = "";
        public string Password { get; set; } = "";
        public string? AccessCode { get; set; } = "";
    }
}