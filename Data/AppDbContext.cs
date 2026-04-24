using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<AppUser> Users { get; set; }
    public DbSet<AdminUser> AdminUsers { get; set; }
    public DbSet<Document> Documents { get; set; }
    public DbSet<DocumentChunk> DocumentChunks { get; set; }
    public DbSet<Question> Questions { get; set; }
    public DbSet<Answer> Answers { get; set; }
    public DbSet<ChunkUsage> ChunkUsages { get; set; }
    public DbSet<Feedback> Feedbacks { get; set; }
    public DbSet<RetrievalLog> RetrievalLogs { get; set; }
    public DbSet<ProcessingLog> ProcessingLogs { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<AppUser>().HasIndex(u => u.Email).IsUnique();
        modelBuilder.Entity<AdminUser>().HasIndex(a => a.Username).IsUnique();

        modelBuilder.Entity<Document>()
            .HasMany(d => d.Chunks)
            .WithOne(c => c.Document)
            .HasForeignKey(c => c.DocumentId);

        modelBuilder.Entity<Question>()
            .HasOne(q => q.User)
            .WithMany(u => u.Questions)
            .HasForeignKey(q => q.UserId);

        modelBuilder.Entity<Answer>()
            .HasOne(a => a.Question)
            .WithOne(q => q.Answer)
            .HasForeignKey<Answer>(a => a.QuestionId);

        modelBuilder.Entity<ChunkUsage>()
            .HasOne(c => c.Answer)
            .WithMany(a => a.ChunkUsages)
            .HasForeignKey(c => c.AnswerId);

        modelBuilder.Entity<ChunkUsage>()
            .HasOne(c => c.Chunk)
            .WithMany(ch => ch.ChunkUsages)
            .HasForeignKey(c => c.ChunkId);
    }
}
