using Fyp.Data;
using Fyp.Services;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    WebRootPath = Path.Combine("client-web", "dist")
});

builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

builder.Services.AddControllers();
builder.Services.AddCors(options =>
{
    options.AddPolicy("ReactClients", policy =>
    {
        policy
            .WithOrigins(
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8081",
                "http://127.0.0.1:8081")
            .AllowAnyHeader()
            .AllowAnyMethod()
            .AllowCredentials();
    });
});

var conn = builder.Configuration.GetConnectionString("DefaultConnection");
if (string.IsNullOrWhiteSpace(conn))
{
    throw new Exception("Connection string 'DefaultConnection' is missing in appsettings.json.");
}

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseMySql(conn, ServerVersion.AutoDetect(conn)));

builder.Services.AddDataProtection()
    .PersistKeysToFileSystem(new DirectoryInfo(Path.Combine(builder.Environment.ContentRootPath, "DataProtectionKeys")));

builder.Services.AddDistributedMemoryCache();
builder.Services.AddSession(options =>
{
    options.IdleTimeout = TimeSpan.FromHours(6);
    options.Cookie.HttpOnly = true;
    options.Cookie.IsEssential = true;
});

builder.Services.AddHttpClient();

builder.Services.AddScoped<PasswordService>();
builder.Services.AddScoped<StudentAuthService>();
builder.Services.AddScoped<AdminAuthService>();
builder.Services.AddScoped<DocumentService>();
builder.Services.AddScoped<TextExtractService>();
builder.Services.AddScoped<ChunkingService>();
builder.Services.AddScoped<MlEmbeddingService>();
builder.Services.AddScoped<QdrantService>();
builder.Services.AddScoped<RagService>();
builder.Services.AddScoped<AnalyticsService>();
builder.Services.AddHostedService<Fyp.Services.StartupOrchestratorService>();

var app = builder.Build();


if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/api/error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseCors("ReactClients");
app.UseSession();

app.MapControllers();
app.MapGet("/api/error", () => Results.Problem("An unexpected error occurred."));
app.MapFallbackToFile("index.html");

app.Run();
