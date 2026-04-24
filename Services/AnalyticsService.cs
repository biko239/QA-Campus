using Fyp.Data;
using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Services
{
    public class AnalyticsService
    {
        private readonly AppDbContext _db;

        public AnalyticsService(AppDbContext db)
        {
            _db = db;
        }

        public async Task<DashboardStatsViewModel> GetDashboardStatsAsync()
        {
            return new DashboardStatsViewModel
            {
                Students = await _db.Users.CountAsync(),
                Documents = await _db.Documents.CountAsync(),
                Questions = await _db.Questions.CountAsync(),
                Answers = await _db.Answers.CountAsync()
            };
        }
    }
}