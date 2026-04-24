namespace Fyp.Models
{
    public class Document
    {
        public int Id { get; set; }

        public string Title { get; set; } = "";
        public string OriginalFileName { get; set; } = "";
        public string StoredFileName { get; set; } = "";

        public string Department { get; set; } = "";
        public string Course { get; set; } = "";
        public string Category { get; set; } = "";
        public string Description { get; set; } = "";
        public string Tags { get; set; } = "";

        public bool IsPublic { get; set; }

        public string FileType { get; set; } = "";
        public long FileSize { get; set; }

        public string Status { get; set; } = "UPLOADED";
        public string ProcessingMethod { get; set; } = "PDF_RAG_PIPELINE";

        public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
        public DateTime? ProcessingStartedAt { get; set; }
        public DateTime? ProcessingCompletedAt { get; set; }

        public List<DocumentChunk> Chunks { get; set; } = new();
    }
}