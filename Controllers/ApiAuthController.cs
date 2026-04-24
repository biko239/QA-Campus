using Fyp.Models;
using Fyp.Services;
using Microsoft.AspNetCore.Mvc;

namespace Fyp.Controllers;

[ApiController]
[Route("api/auth")]
public class ApiAuthController : ControllerBase
{
    private readonly StudentAuthService _studentAuth;
    private readonly AdminAuthService _adminAuth;

    public ApiAuthController(StudentAuthService studentAuth, AdminAuthService adminAuth)
    {
        _studentAuth = studentAuth;
        _adminAuth = adminAuth;
    }

    [HttpGet("me")]
    public IActionResult Me()
    {
        var role = HttpContext.Session.GetString("Role");
        if (string.IsNullOrWhiteSpace(role))
            return Unauthorized(new { message = "Not signed in." });

        return Ok(new
        {
            role,
            userId = HttpContext.Session.GetInt32("UserId"),
            adminId = HttpContext.Session.GetInt32("AdminId"),
            name = HttpContext.Session.GetString("Name") ?? HttpContext.Session.GetString("AdminName") ?? ""
        });
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login(LoginRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Identifier) || string.IsNullOrWhiteSpace(request.Password))
            return BadRequest(new { message = "Email/Username and password are required." });

        var student = await _studentAuth.LoginAsync(request.Identifier, request.Password);
        if (student != null)
        {
            HttpContext.Session.SetInt32("UserId", student.Id);
            HttpContext.Session.SetString("Role", "Student");
            HttpContext.Session.SetString("Name", $"{student.FirstName} {student.LastName}".Trim());

            return Ok(new { role = "Student", name = $"{student.FirstName} {student.LastName}".Trim(), userId = student.Id });
        }

        var admin = await _adminAuth.LoginAsync(request.Identifier, request.Password, request.AccessCode);
        if (admin != null)
        {
            HttpContext.Session.SetInt32("AdminId", admin.Id);
            HttpContext.Session.SetString("Role", "Admin");
            HttpContext.Session.SetString("AdminName", admin.Username);

            return Ok(new { role = "Admin", name = admin.Username, adminId = admin.Id });
        }

        return Unauthorized(new { message = "Invalid credentials." });
    }

    [HttpPost("students/register")]
    public async Task<IActionResult> RegisterStudent(StudentRegisterRequest request)
    {
        if (!request.TermsAccepted)
            return BadRequest(new { message = "You must accept the terms and conditions." });

        if (string.IsNullOrWhiteSpace(request.Password) || string.IsNullOrWhiteSpace(request.ConfirmPassword))
            return BadRequest(new { message = "Password and confirm password are required." });

        if (request.Password != request.ConfirmPassword)
            return BadRequest(new { message = "Passwords do not match." });

        var model = new AppUser
        {
            FirstName = request.FirstName ?? "",
            LastName = request.LastName ?? "",
            StudentNumber = request.StudentNumber ?? "",
            Email = request.Email ?? "",
            Department = request.Department ?? "",
            TermsAccepted = request.TermsAccepted
        };

        var result = await _studentAuth.RegisterAsync(model, request.Password);
        if (!result.ok)
            return BadRequest(new { message = result.message });

        return Ok(new { message = "Account created successfully. Please log in." });
    }

    [HttpPost("admins/register")]
    public async Task<IActionResult> RegisterAdmin(AdminRegisterRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Username) ||
            string.IsNullOrWhiteSpace(request.Password) ||
            string.IsNullOrWhiteSpace(request.ConfirmPassword) ||
            string.IsNullOrWhiteSpace(request.AccessCode) ||
            string.IsNullOrWhiteSpace(request.ConfirmAccessCode))
        {
            return BadRequest(new { message = "All fields are required." });
        }

        if (request.Password != request.ConfirmPassword)
            return BadRequest(new { message = "Passwords do not match." });

        if (request.AccessCode != request.ConfirmAccessCode)
            return BadRequest(new { message = "Access codes do not match." });

        var result = await _adminAuth.RegisterAsync(request.Username, request.Password, request.AccessCode);
        if (!result.ok)
            return BadRequest(new { message = result.message });

        return Ok(new { message = "Admin account created successfully. Please log in." });
    }

    [HttpPost("logout")]
    public IActionResult Logout()
    {
        HttpContext.Session.Clear();
        return Ok(new { message = "Signed out." });
    }
}

public record LoginRequest(string Identifier, string Password, string? AccessCode);
public record StudentRegisterRequest(
    string? FirstName,
    string? LastName,
    string? StudentNumber,
    string? Email,
    string? Department,
    string Password,
    string ConfirmPassword,
    bool TermsAccepted);
public record AdminRegisterRequest(
    string? Username,
    string? Password,
    string? ConfirmPassword,
    string? AccessCode,
    string? ConfirmAccessCode);
