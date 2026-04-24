using System.Text;
using UglyToad.PdfPig;

namespace Fyp.Services
{
    public class TextExtractService
    {
        public async Task<string> ExtractAsync(string filePath)
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();

            if (ext == ".txt")
            {
                return await File.ReadAllTextAsync(filePath);
            }

            if (ext == ".pdf")
            {
                return await Task.Run(() =>
                {
                    var sb = new StringBuilder();

                    using (var document = PdfDocument.Open(filePath))
                    {
                        foreach (var page in document.GetPages())
                        {
                            sb.AppendLine(page.Text);
                            sb.AppendLine();
                        }
                    }

                    return sb.ToString();
                });
            }

            throw new Exception($"Unsupported file type: {ext}");
        }
    }
}