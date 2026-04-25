using System.Text;
using System.Text.RegularExpressions;
using UglyToad.PdfPig;

namespace Fyp.Services
{
    public class TextExtractService
    {
        public async Task<string> ExtractAsync(string filePath)
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();

            if (ext == ".txt")
                return NormalizeExtractedText(await File.ReadAllTextAsync(filePath));

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

                    return NormalizeExtractedText(sb.ToString());
                });
            }

            throw new Exception($"Unsupported file type: {ext}");
        }

        private static string NormalizeExtractedText(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return "";

            var text = value.Replace("\r\n", "\n").Replace('\r', '\n');

            text = Regex.Replace(text, @"-\s*\n\s*", "");
            text = Regex.Replace(text, @"(?<=\p{Ll})(?=\p{Lu})", " ");
            text = Regex.Replace(text, @"(?<=\p{L})(?=\d)", " ");
            text = Regex.Replace(text, @"(?<=\d)(?=\p{L})", " ");
            text = Regex.Replace(text, @"(?<=\p{L})(?=(Article|Section|Chapter|Titre|Chapitre)\s+\d)", " ", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"[ \t]+", " ");
            text = Regex.Replace(text, @" *\n+ *", "\n");
            text = Regex.Replace(text, @"\n{3,}", "\n\n");

            return text.Trim();
        }
    }
}
