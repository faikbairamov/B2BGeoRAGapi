const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { pipeline } = require('@xenova/transformers');
const { Pinecone } = require('@pinecone-database/pinecone');
const pdfParse = require('pdf-parse');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');

// Configuration Constants
const BATCH_SIZE = 100;

// Pinecone Configuration
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = "georag1";
const PINECONE_INDEX_REGION = "us-east-1";

// Model Configuration
const EMBEDDING_MODEL_NAME = "Xenova/bge-m3";
const EMBEDDING_DIMENSION = 1024;
const SIMILARITY_METRIC = "cosine";

// Semantic Chunking Configuration
const CHUNK_SIZE = 1000; // Target chunk size in characters
const CHUNK_OVERLAP = 200; // Overlap between chunks for context preservation
const SEPARATORS = ['\n\n', '\n', '.', '!', '?', ';', ':', ' ', '']; // Priority order for splitting

// Initialize Pinecone client
const pineconeClient = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

// ===== HELPER FUNCTIONS =====

async function ensurePineconeIndexExists() {
  console.log("🔍 Checking if Pinecone index exists...");
  const indexList = await pineconeClient.listIndexes();
  const indexExists = indexList.indexes?.some(
    (index) => index.name === PINECONE_INDEX_NAME
  );

  if (!indexExists) {
    console.log(`📝 Creating Pinecone index '${PINECONE_INDEX_NAME}'...`);
    await pineconeClient.createIndex({
      name: PINECONE_INDEX_NAME,
      dimension: EMBEDDING_DIMENSION,
      metric: SIMILARITY_METRIC,
      spec: {
        serverless: {
          cloud: "aws",
          region: PINECONE_INDEX_REGION,
        },
      },
    });
    console.log(`✅ Pinecone index '${PINECONE_INDEX_NAME}' created successfully`);
    
    // Wait a bit for index to be ready
    await new Promise(resolve => setTimeout(resolve, 10000));
  } else {
    console.log(`✅ Pinecone index '${PINECONE_INDEX_NAME}' already exists`);
  }
}

async function createSemanticChunks(textContent, filename) {
  console.log(`✂️ Creating semantic chunks using LangChain RecursiveCharacterTextSplitter...`);
  console.log(`📊 Chunk size: ${CHUNK_SIZE}, Overlap: ${CHUNK_OVERLAP}`);

  // Initialize the semantic text splitter
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    separators: SEPARATORS,
    keepSeparator: true, // Keep separators to maintain context
    lengthFunction: (text) => text.length, // Use character count as length function
  });

  try {
    // Split the text into semantic chunks
    const textChunks = await textSplitter.splitText(textContent);
    
    // Transform into our expected format with metadata
    const formattedChunks = textChunks.map((chunkText, index) => {
      const startPosition = index === 0 ? 0 : 
        textContent.indexOf(chunkText, index > 0 ? textContent.indexOf(textChunks[index - 1]) + textChunks[index - 1].length : 0);
      
      return {
        id: uuidv4(),
        text: chunkText.trim(),
        wordCount: chunkText.trim().split(/\s+/).length,
        charCount: chunkText.trim().length,
        startPosition: startPosition >= 0 ? startPosition : index * CHUNK_SIZE, // Fallback if indexOf fails
        chunkIndex: index,
        filename: filename,
        chunkType: 'semantic', // Mark as semantic chunk
      };
    });

    console.log(`✅ Created ${formattedChunks.length} semantic chunks`);
    console.log(`📈 Average chunk size: ${Math.round(formattedChunks.reduce((sum, chunk) => sum + chunk.charCount, 0) / formattedChunks.length)} characters`);
    console.log(`📏 Chunk size range: ${Math.min(...formattedChunks.map(c => c.charCount))} - ${Math.max(...formattedChunks.map(c => c.charCount))} characters`);
    
    return formattedChunks;
  } catch (error) {
    console.error(`❌ Error creating semantic chunks:`, error);
    throw new Error(`Semantic chunking failed: ${error.message}`);
  }
}

async function generateEmbeddingsForChunks(textChunks) {
  console.log(`🧠 Initializing embedding model '${EMBEDDING_MODEL_NAME}'...`);
  const embeddingPipeline = await pipeline(
    "feature-extraction",
    EMBEDDING_MODEL_NAME
  );
  console.log("✅ Embedding model loaded successfully");

  const embeddingVectors = [];
  const totalChunks = textChunks.length;

  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    if (chunkIndex % 10 === 0 || chunkIndex === totalChunks - 1) {
      console.log(`⚡ Processing chunk ${chunkIndex + 1} of ${totalChunks}...`);
    }
    
    try {
      const embeddingOutput = await embeddingPipeline(
        textChunks[chunkIndex].text,
        {
          pooling: "cls",
          normalize: true,
        }
      );
      embeddingVectors.push(embeddingOutput.data);
    } catch (error) {
      console.error(`❌ Error generating embedding for chunk ${chunkIndex + 1}:`, error);
      throw new Error(`Embedding generation failed at chunk ${chunkIndex + 1}: ${error.message}`);
    }
  }

  console.log(`✅ Generated ${embeddingVectors.length} embedding vectors`);
  return embeddingVectors;
}

async function uploadChunksToPinecone(textChunks, embeddingVectors, userId, filename) {
  console.log(`💾 Preparing to upload ${textChunks.length} vectors to Pinecone...`);
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
  const vectorsToUpload = [];

  for (let chunkIndex = 0; chunkIndex < textChunks.length; chunkIndex++) {
    const currentChunk = textChunks[chunkIndex];
    const currentEmbedding = embeddingVectors[chunkIndex];
    
    vectorsToUpload.push({
      id: currentChunk.id,
      values: Array.from(currentEmbedding),
      metadata: {
        userId: userId,
        filename: filename,
        text: currentChunk.text,
        wordCount: currentChunk.wordCount,
        charCount: currentChunk.charCount,
        startPosition: currentChunk.startPosition,
        chunkIndex: currentChunk.chunkIndex,
        chunkId: currentChunk.id,
        chunkType: currentChunk.chunkType,
        uploadedAt: new Date().toISOString(),
        // Additional semantic chunking metadata
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP,
      },
    });
  }

  const totalBatches = Math.ceil(vectorsToUpload.length / BATCH_SIZE);
  console.log(`📦 Uploading in ${totalBatches} batches of ${BATCH_SIZE}...`);

  for (let batchIndex = 0; batchIndex < vectorsToUpload.length; batchIndex += BATCH_SIZE) {
    const currentBatch = vectorsToUpload.slice(batchIndex, batchIndex + BATCH_SIZE);
    const batchNumber = Math.floor(batchIndex / BATCH_SIZE) + 1;
    
    try {
      await pineconeIndex.upsert(currentBatch);
      console.log(`✅ Uploaded batch ${batchNumber} of ${totalBatches}`);
    } catch (error) {
      console.error(`❌ Failed to upload batch ${batchNumber}:`, error.message);
      throw new Error(`Batch upload failed: ${error.message}`);
    }
  }
  
  console.log(`🎉 Successfully uploaded all ${vectorsToUpload.length} vectors to Pinecone`);
}

// ===== SEARCH FUNCTION =====

async function semanticSearch(query, userId, topK = 5) {
  console.log(`🔍 Performing semantic search for query: "${query}"`);
  
  try {
    // Generate embedding for the query
    console.log(`🧠 Generating embedding for search query...`);
    const embeddingPipeline = await pipeline("feature-extraction", EMBEDDING_MODEL_NAME);
    const queryEmbeddingOutput = await embeddingPipeline(query, {
      pooling: "cls",
      normalize: true,
    });
    const queryEmbedding = Array.from(queryEmbeddingOutput.data);

    // Search in Pinecone
    const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
      vector: queryEmbedding,
      topK: topK,
      filter: { userId: userId }, // Filter by user
      includeMetadata: true,
      includeValues: false,
    });

    console.log(`✅ Found ${searchResults.matches.length} relevant chunks`);
    
    return searchResults.matches.map(match => ({
      id: match.id,
      score: match.score,
      text: match.metadata.text,
      filename: match.metadata.filename,
      chunkIndex: match.metadata.chunkIndex,
      wordCount: match.metadata.wordCount,
      charCount: match.metadata.charCount,
      chunkType: match.metadata.chunkType,
    }));
    
  } catch (error) {
    console.error(`❌ Error performing semantic search:`, error);
    throw new Error(`Semantic search failed: ${error.message}`);
  }
}

// ===== MAIN CONTROLLER =====

exports.uploadFiles = async (req, res) => {
  console.log('--- /api/query/ UPLOAD REQUEST ---');
  console.log('Headers:', req.headers);
  console.log('Body:', req.body);
  console.log('Files:', req.files);

  try {
    console.log("🚀 Starting PDF processing with semantic chunking RAG pipeline...");
    
    // Step 1: Ensure Pinecone index exists
    await ensurePineconeIndexExists();
    
    const results = [];
    const userId = req.user?._id.toString() || "default_user"; 
    console.log(`👤 Processing files for user: ${userId}`);

    // Step 2: Process each uploaded PDF file
    for (const file of req.files) {
      console.log(`📄 Processing file: ${file.originalname}`);
      
      const dataBuffer = fs.readFileSync(file.path);
      const data = await pdfParse(dataBuffer);

      // Clean the extracted text
      const cleanedText = data.text
        .replace(/\s+/g, ' ') // Replace multiple whitespace with single space
        .replace(/\n\s*\n/g, '\n\n') // Normalize paragraph breaks
        .trim();
      
      console.log(`📝 Extracted ${cleanedText.length} characters from ${file.originalname}`);

      // Step 3: Create semantic chunks using LangChain
      const textChunks = await createSemanticChunks(cleanedText, file.originalname);

      // Step 4: Generate embeddings
      const embeddingVectors = await generateEmbeddingsForChunks(textChunks);

      // Step 5: Upload to Pinecone
      await uploadChunksToPinecone(textChunks, embeddingVectors, userId, file.originalname);

      results.push({
        filename: file.originalname,
        textLength: cleanedText.length,
        chunksCreated: textChunks.length,
        vectorsUploaded: embeddingVectors.length,
        chunkingMethod: 'semantic',
        avgChunkSize: Math.round(textChunks.reduce((sum, chunk) => sum + chunk.charCount, 0) / textChunks.length),
        chunkSizeRange: {
          min: Math.min(...textChunks.map(c => c.charCount)),
          max: Math.max(...textChunks.map(c => c.charCount))
        },
        status: 'success'
      });

      // Clean up temp file
      fs.unlinkSync(file.path);
      console.log(`🗑️ Cleaned up temporary file: ${file.path}`);
    }

    console.log("🎉 All files processed successfully with semantic chunking!");

    res.status(200).json({
      success: true,
      message: 'PDF files processed with semantic chunking and indexed successfully',
      results: results,
      totalFiles: req.files.length,
      chunkingConfig: {
        method: 'semantic',
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP,
        separators: SEPARATORS,
      }
    });

  } catch (err) {
    console.error('❌ Error processing PDF files:', err);
    
    // Clean up any remaining temp files
    if (req.files) {
      req.files.forEach(file => {
        try {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
            console.log(`🗑️ Cleaned up temp file: ${file.path}`);
          }
        } catch (cleanupErr) {
          console.error('Error cleaning up temp file:', cleanupErr);
        }
      });
    }

    res.status(500).json({ 
      success: false,
      error: 'Failed to process PDF files with semantic chunking',
      message: err.message 
    });
  }
};

// Export the search function as well
exports.search = async (req, res) => {
  try {
    const { query, topK = 5 } = req.body;
    const userId = req.user?._id.toString() || "default_user";
    
    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query is required'
      });
    }
    
    const searchResults = await semanticSearch(query, userId, topK);
    
    res.status(200).json({
      success: true,
      query: query,
      results: searchResults,
      resultCount: searchResults.length
    });
    
  } catch (err) {
    console.error('❌ Error performing search:', err);
    res.status(500).json({
      success: false,
      error: 'Search failed',
      message: err.message
    });
  }
};