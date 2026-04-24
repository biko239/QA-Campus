using Fyp.Data;
using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Services
{
    public static class AdminSeedService
    {
        public static async Task SeedDefaultAdminAsync(IServiceProvider services)
        {
            using var scope = services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            var pwd = scope.ServiceProvider.GetRequiredService<PasswordService>();

            await db.Database.MigrateAsync();

            bool hasAdmin = await db.AdminUsers.AnyAsync();
            if (hasAdmin) return;

            var admin = new AdminUser
            {
                Username = "admin",
                PasswordHash = pwd.Hash("admin123"),
                AccessCodeHash = pwd.Hash("9999"),
                CreatedAt = DateTime.UtcNow
            };

            db.AdminUsers.Add(admin);
            await db.SaveChangesAsync();
        }
    }
}