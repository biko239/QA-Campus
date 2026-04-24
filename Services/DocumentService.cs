using Fyp.Data;
using Fyp.Models;
using Microsoft.EntityFrameworkCore;

namespace Fyp.Services
{
    public class DocumentService
    {
        private readonly AppDbContext _db;
        private readonly IConfiguration _cfg;

        public DocumentService(AppDbContext db, IConfiguration cfg)
        {
            _db = db;
            _cfg = cfg;
        }

        public string UploadFolder()
        {
            var folder = _cfg["Uploads:Folder"] ?? "Uploads";
            Directory.CreateDirectory(folder);
            return folder;
        }

        public async Task<Document> CreateDocumentAsync(
            IFormFile file,
            string title,
            string department,
            string course,
            string category,
            string description,
            string tags,
            bool isPublic)
        {
            if (file == null || file.Length == 0)
                throw new Exception("No file uploaded.");

            var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
            if (ext != ".pdf" && ext != ".txt")
                throw new Exception("Only PDF and TXT files are supported.");

            var folder = UploadFolder();
            var stored = $"{Guid.NewGuid():N}_{Path.GetFileName(file.FileName)}";
            var path = Path.Combine(folder, stored);

            using (var stream = File.Create(path))
            {
                await file.CopyToAsync(stream);
            }

            var doc = new Document
            {
                Title = string.IsNullOrWhiteSpace(title)
                    ? Path.GetFileNameWithoutExtension(file.FileName)
                    : title.Trim(),
                OriginalFileName = file.FileName,
                StoredFileName = stored,
                Department = department ?? "",
                Course = course ?? "",
                Category = category ?? "",
                Description = description ?? "",
                Tags = tags ?? "",
                IsPublic = isPublic,
                FileType = ext,
                FileSize = file.Length,
                Status = "UPLOADED",
                ProcessingMethod = "PDF_RAG_PIPELINE",
                UploadedAt = DateTime.UtcNow
            };

            _db.Documents.Add(doc);
            await _db.SaveChangesAsync();

            return doc;
        }

        public async Task<List<Document>> ListAsync()
        {
            return await _db.Documents
                .OrderByDescending(d => d.UploadedAt)
                .ToListAsync();
        }

        public async Task<Document?> GetAsync(int id)
        {
            return await _db.Documents
                .Include(d => d.Chunks)
                .FirstOrDefaultAsync(d => d.Id == id);
        }

        public async Task DeleteAsync(int id)
        {
            var doc = await _db.Documents
                .Include(d => d.Chunks)
                .FirstOrDefaultAsync(d => d.Id == id);

            if (doc == null) return;

            var path = Path.Combine(UploadFolder(), doc.StoredFileName);
            if (File.Exists(path))
                File.Delete(path);

            _db.DocumentChunks.RemoveRange(doc.Chunks);
            _db.Documents.Remove(doc);

            await _db.SaveChangesAsync();
        }
    }
}