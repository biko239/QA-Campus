using System.Security.Cryptography;
using System.Text;

namespace Fyp.Services;

public class PasswordService
{
    public string Hash(string input)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(input ?? string.Empty));
        return Convert.ToHexString(bytes);
    }
}
