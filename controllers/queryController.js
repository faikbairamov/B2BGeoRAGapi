const { pipeline } = require('@xenova/transformers');
const { Pinecone } = require('@pinecone-database/pinecone');
const OpenAI = require('openai');

// Configuration Constants (should match your upload controller)
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = "georag1";
const EMBEDDING_MODEL_NAME = "Xenova/bge-m3";

// Initialize clients
const pineconeClient = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY 
});

// ===== HELPER FUNCTIONS =====

async function generateQueryEmbedding(queryText) {
  console.log(`🧠 Embedding query: "${queryText}"...`);
  const embeddingPipeline = await pipeline(
    "feature-extraction",
    EMBEDDING_MODEL_NAME
  );
  const queryEmbeddingOutput = await embeddingPipeline(queryText, {
    pooling: "cls",
    normalize: true,
  });
  console.log("✅ Query embedding generated");
  return queryEmbeddingOutput.data;
}

async function answerWithGPT4O(question, topChunks) {
   console.log("topChunks:", topChunks);
  const context = topChunks
    .map((chunk, index) => `Context ${index + 1}: ${chunk.fullText}`)
    .join("\n\n");
    
  const prompt = `
You are an expert assistant. Use ONLY the following context to answer the user's question.

${context}

Question: ${question}
Answer:`;


console.log(`prompt: ${prompt}`);
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 512,
    temperature: 0.2,
  });
  console.log('content, ', completion.choices[0].message.content);
  
  return completion.choices[0].message.content.trim();
}

async function searchSimilarChunks(queryEmbedding, userId, maxResults = 5) {
  console.log(`🔍 Querying Pinecone index for ${maxResults} similar chunks...`);
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
  
  const searchResults = await pineconeIndex.query({
    vector: Array.from(queryEmbedding),
    topK: maxResults * 2, // Fetch more to allow for filtering
    includeMetadata: true,
  });

  // Filter matches by userId
  const filteredMatches = (searchResults.matches || [])
    .filter((match) => match.metadata?.userId === userId)
    .slice(0, maxResults);

  return filteredMatches.map((match, index) => {
    const similarityScore = (match.score * 100).toFixed(2);
    const previewText = match.metadata?.text?.slice(0, 300) || "No text available";
    
    console.log(`🔹 Rank ${index + 1} | Similarity: ${similarityScore}%`);
    console.log(`📄 Filename: ${match.metadata?.filename}`);
    console.log(`📝 Preview: ${previewText}...`);
    console.log("─".repeat(80));
    
    return {
      id: match.id,
      similarity: match.score,
      preview: previewText,
      fullText: match.metadata?.text || "",
      filename: match.metadata?.filename,
      userId: match.metadata?.userId,
      chunkId: match.metadata?.chunkId,
    };
  });
}

// ===== MAIN CONTROLLER =====

exports.sendPrompt = async (req, res) => {
  console.log('--- /api/chat/sendPrompt REQUEST ---');
  console.log('Body:', req.body);
  console.log('User:', req.user?._id);

  try {
    // Extract query and options from request
    const { 
      query, 
      maxResults = 5,
      includeMetadata = true 
    } = req.body;

    // Validate input
    if (!query || query.trim() === '') {
      return res.status(400).json({
        success: false,
        error: 'Query is required',
        message: 'Please provide a valid search query'
      });
    }

    // Get userId from authenticated user
    const userId = req.user?._id.toString();
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required',
        message: 'User must be authenticated to send prompts'
      });
    }

    console.log(`🔎 Processing query for user ${userId}: "${query}"`);

    // Step 1: Generate embedding for the query
    const queryEmbedding = await generateQueryEmbedding(query);

    // Step 2: Search for similar chunks in vector database
    const similarChunks = await searchSimilarChunks(queryEmbedding, userId, maxResults);

    if (similarChunks.length === 0) {
      console.log("❌ No relevant chunks found for this user");
      return res.status(200).json({
        success: true,
        query: query,
        answer: "I couldn't find any relevant information in your uploaded documents to answer this question. Please make sure you have uploaded documents that contain information related to your query.",
        sources: [],
        metadata: {
          userId: userId,
          chunksFound: 0,
          processingTime: new Date().toISOString()
        }
      });
    }

    // Step 3: Generate AI answer using the retrieved context
    console.log(`💡 Generating AI answer using ${similarChunks.length} relevant chunks...`);
    const aiAnswer = await answerWithGPT4O(query, similarChunks);

    // Step 4: Prepare response
    const response = {
      success: true,
      query: query,
      answer: aiAnswer,
      sources: includeMetadata ? similarChunks.map(chunk => ({
        filename: chunk.filename,
        similarity: chunk.similarity,
        preview: chunk.preview,
        chunkId: chunk.chunkId
      })) : [],
      metadata: {
        userId: userId,
        chunksFound: similarChunks.length,
        maxSimilarity: similarChunks[0]?.similarity || 0,
        processingTime: new Date().toISOString(),
        model: "gpt-4o",
        embeddingModel: EMBEDDING_MODEL_NAME
      }
    };

    console.log("✅ Query processed successfully");
    res.status(200).json(response);

  } catch (error) {
    console.error('❌ Error processing prompt:', error);
    
    res.status(500).json({
      success: false,
      error: 'Failed to process prompt',
      message: error.message,
      metadata: {
        userId: req.user?._id?.toString() || 'unknown',
        processingTime: new Date().toISOString()
      }
    });
  }
};