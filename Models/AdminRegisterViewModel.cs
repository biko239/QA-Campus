namespace Fyp.Models
{
    public class AdminRegisterViewModel
    {
        public string Username { get; set; } = "";
        public string Password { get; set; } = "";
        public string ConfirmPassword { get; set; } = "";
        public string AccessCode { get; set; } = "";
        public string ConfirmAccessCode { get; set; } = "";
    }
}