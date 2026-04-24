using Fyp.Data;
using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Services
{
    public class AdminAuthService
    {
        private readonly AppDbContext _db;
        private readonly PasswordService _pwd;

        public AdminAuthService(AppDbContext db, PasswordService pwd)
        {
            _db = db;
            _pwd = pwd;
        }

        public async Task<(bool ok, string message)> RegisterAsync(string? username, string? password, string? accessCode)
        {
            if (string.IsNullOrWhiteSpace(username) ||
                string.IsNullOrWhiteSpace(password) ||
                string.IsNullOrWhiteSpace(accessCode))
            {
                return (false, "Username, password, and access code are required.");
            }

            string normalizedUsername = username.Trim().ToLower();

            bool exists = await _db.AdminUsers.AnyAsync(a => a.Username.ToLower() == normalizedUsername);
            if (exists)
            {
                return (false, "This admin username already exists.");
            }

            var admin = new AdminUser
            {
                Username = normalizedUsername,
                PasswordHash = _pwd.Hash(password),
                AccessCodeHash = _pwd.Hash(accessCode),
                CreatedAt = DateTime.UtcNow
            };

            _db.AdminUsers.Add(admin);
            await _db.SaveChangesAsync();

            return (true, "Admin account created successfully.");
        }

        public async Task<AdminUser?> LoginAsync(string? username, string? password, string? accessCode)
        {
            if (string.IsNullOrWhiteSpace(username) ||
                string.IsNullOrWhiteSpace(password) ||
                string.IsNullOrWhiteSpace(accessCode))
            {
                return null;
            }

            string normalizedUsername = username.Trim().ToLower();
            string passwordHash = _pwd.Hash(password);
            string accessCodeHash = _pwd.Hash(accessCode);

            return await _db.AdminUsers.FirstOrDefaultAsync(a =>
                a.Username.ToLower() == normalizedUsername &&
                a.PasswordHash == passwordHash &&
                a.AccessCodeHash == accessCodeHash);
        }
    }
}