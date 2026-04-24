using Fyp.Data;
using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Services
{
    public class StudentAuthService
    {
        private readonly AppDbContext _db;
        private readonly PasswordService _pwd;

        public StudentAuthService(AppDbContext db, PasswordService pwd)
        {
            _db = db;
            _pwd = pwd;
        }

        public async Task<(bool ok, string message)> RegisterAsync(AppUser model, string password)
        {
            if (model == null)
                return (false, "Invalid registration data.");

            if (string.IsNullOrWhiteSpace(model.Email))
                return (false, "Email is required.");

            if (string.IsNullOrWhiteSpace(password))
                return (false, "Password is required.");

            string normalizedEmail = model.Email.Trim().ToLower();

            bool exists = await _db.Users.AnyAsync(u => u.Email == normalizedEmail);
            if (exists)
                return (false, "This email already exists.");

            AppUser user = new AppUser
            {
                FirstName = model.FirstName ?? "",
                LastName = model.LastName ?? "",
                StudentNumber = model.StudentNumber ?? "",
                Email = normalizedEmail,
                Department = model.Department ?? "",
                TermsAccepted = model.TermsAccepted,
                PasswordHash = _pwd.Hash(password),
                Role = "Student",
                CreatedAt = DateTime.UtcNow
            };

            _db.Users.Add(user);
            await _db.SaveChangesAsync();

            return (true, "Registered successfully.");
        }

        public async Task<AppUser?> LoginAsync(string? email, string? password)
        {
            if (string.IsNullOrWhiteSpace(email) || string.IsNullOrWhiteSpace(password))
                return null;

            string normalizedEmail = email.Trim().ToLower();
            string hash = _pwd.Hash(password);

            return await _db.Users.FirstOrDefaultAsync(u =>
                u.Email == normalizedEmail &&
                u.PasswordHash == hash &&
                u.Role == "Student");
        }
    }
}