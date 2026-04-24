namespace Fyp.Models;

public class ChunkUsage
{
    public int Id { get; set; }
    public int AnswerId { get; set; }
    public Answer? Answer { get; set; }
    public int ChunkId { get; set; }
    public DocumentChunk? Chunk { get; set; }
    public float Score { get; set; }
}
