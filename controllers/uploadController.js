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
const PINECONE_INDEX_NAME = "vectormind";
const PINECONE_INDEX_REGION = "us-east-1";

// Model Configuration
const EMBEDDING_MODEL_NAME = "Xenova/bge-m3";
const EMBEDDING_DIMENSION = 1024;
const SIMILARITY_METRIC = "cosine";

// Semantic Chunking Configuration - Optimized for extracted text
const CHUNK_SIZE = 800; // Slightly smaller for better sentence completion
const CHUNK_OVERLAP = 150; // Reduced overlap to avoid too much repetition
const SEPARATORS = [
  "\n\n",    // Double newlines (paragraphs) - highest priority
  "\n",      // Single newlines
  ". ",      // Sentence endings with space
  "! ",      // Exclamations with space  
  "? ",      // Questions with space
  "; ",      // Semicolons with space
  ", ",      // Commas with space (be more selective)
  " ",       // Spaces
  ""         // Character level as last resort
];

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

async function createSemanticChunks(textContent) {
  console.log(`✂️ Creating semantic chunks using LangChain RecursiveCharacterTextSplitter...`);
  console.log(`📊 Target chunk size: ${CHUNK_SIZE} characters with ${CHUNK_OVERLAP} overlap`);

  // Initialize the semantic text splitter
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    separators: SEPARATORS,
    lengthFunction: (text) => text.length,
  });

  try {
    // Split the text into semantic chunks
    const documents = await textSplitter.createDocuments([textContent]);
    
    // Convert LangChain documents to our chunk format with quality validation
    const textChunks = documents.map((doc, index) => {
      const chunkText = doc.pageContent;
      const wordCount = chunkText.split(/\s+/).filter(word => word.length > 0).length;
      
      // Calculate quality metrics
      const punctuationRatio = (chunkText.match(/[.,!?;]/g) || []).length / chunkText.length;
      const avgWordLength = chunkText.replace(/[^\w\s]/g, '').split(/\s+/).reduce((sum, word) => sum + word.length, 0) / wordCount;
      const endsWithCompleteWord = /\w$/.test(chunkText.trim());
      
      return {
        id: uuidv4(),
        text: chunkText,
        wordCount: wordCount,
        charCount: chunkText.length,
        chunkIndex: index,
        quality: {
          punctuationRatio: Math.round(punctuationRatio * 1000) / 1000,
          avgWordLength: Math.round(avgWordLength * 10) / 10,
          endsWithCompleteWord: endsWithCompleteWord,
          qualityScore: endsWithCompleteWord ? (avgWordLength > 2 ? 'good' : 'fair') : 'poor'
        },
        metadata: doc.metadata || {}
      };
    });

    // Filter out very poor quality chunks
    const filteredChunks = textChunks.filter(chunk => {
      if (chunk.wordCount < 5 || chunk.quality.punctuationRatio > 0.3) {
        console.warn(`⚠️ Removing poor quality chunk: "${chunk.text.substring(0, 50)}..."`);
        return false;
      }
      return true;
    });

    console.log(`✅ Created ${filteredChunks.length} semantic chunks (filtered out ${textChunks.length - filteredChunks.length} poor quality chunks)`);
    console.log(`📈 Average chunk size: ${Math.round(filteredChunks.reduce((sum, chunk) => sum + chunk.charCount, 0) / filteredChunks.length)} characters`);
    console.log(`📈 Average words per chunk: ${Math.round(filteredChunks.reduce((sum, chunk) => sum + chunk.wordCount, 0) / filteredChunks.length)} words`);
    
    // Log quality distribution
    const qualityCount = filteredChunks.reduce((acc, chunk) => {
      acc[chunk.quality.qualityScore] = (acc[chunk.quality.qualityScore] || 0) + 1;
      return acc;
    }, {});
    console.log(`📊 Quality distribution:`, qualityCount);
    
    // Log chunk size distribution for analysis
    const chunkSizes = filteredChunks.map(chunk => chunk.charCount);
    const minSize = Math.min(...chunkSizes);
    const maxSize = Math.max(...chunkSizes);
    console.log(`📏 Chunk size range: ${minSize} - ${maxSize} characters`);

    return filteredChunks;
  } catch (error) {
    console.error('❌ Error creating semantic chunks:', error);
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
      throw new Error(`Embedding generation failed for chunk ${chunkIndex + 1}: ${error.message}`);
    }
  }

  console.log(`✅ Generated ${embeddingVectors.length} embedding vectors`);
  return embeddingVectors;
}

// Helper function to sanitize metadata for Pinecone
function sanitizeMetadata(metadata) {
  const sanitized = {};
  
  for (const [key, value] of Object.entries(metadata)) {
    if (value === null || value === undefined) {
      continue; // Skip null/undefined values
    }
    
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      sanitized[key] = value;
    } else if (Array.isArray(value) && value.every(item => typeof item === 'string')) {
      sanitized[key] = value; // Array of strings is allowed
    } else if (typeof value === 'object') {
      // Convert complex objects to JSON strings
      try {
        sanitized[`${key}_json`] = JSON.stringify(value);
      } catch (error) {
        console.warn(`⚠️ Could not serialize metadata field '${key}':`, error);
      }
    } else {
      // Convert other types to strings
      sanitized[key] = String(value);
    }
  }
  
  return sanitized;
}

async function uploadChunksToPinecone(textChunks, embeddingVectors, userId, filename) {
  console.log(`💾 Preparing to upload ${textChunks.length} vectors to Pinecone...`);
  const pineconeIndex = pineconeClient.Index(PINECONE_INDEX_NAME);
  const vectorsToUpload = [];

  for (let chunkIndex = 0; chunkIndex < textChunks.length; chunkIndex++) {
    const currentChunk = textChunks[chunkIndex];
    const currentEmbedding = embeddingVectors[chunkIndex];
    
    // Sanitize LangChain metadata for Pinecone compatibility
    const sanitizedLangChainMetadata = sanitizeMetadata(currentChunk.metadata || {});
    
    vectorsToUpload.push({
      id: currentChunk.id,
      values: Array.from(currentEmbedding),
      metadata: {
        userId: userId,
        filename: filename,
        text: currentChunk.text,
        wordCount: currentChunk.wordCount,
        charCount: currentChunk.charCount,
        chunkIndex: currentChunk.chunkIndex,
        chunkId: currentChunk.id,
        ...sanitizedLangChainMetadata,
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
  
  console.log(`🎉 Successfully uploaded all ${vectorsToUpload.length} semantic chunks to Pinecone`);
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
    console.log(`👤 Processing for user: ${userId}`);

    // Step 2: Process each uploaded PDF file
    for (const file of req.files) {
      console.log(`📄 Processing file: ${file.originalname}`);
      
      try {
        const dataBuffer = fs.readFileSync(file.path);
        const data = await pdfParse(dataBuffer);

        // Enhanced text cleaning for Georgian PDF parsing issues
        const cleanedText = data.text
          // First pass: handle newlines and basic structure
          .replace(/\n+/g, ' ')                           // Convert all newlines to spaces first
          .replace(/\s+/g, ' ')                           // Normalize all whitespace
          
          // Second pass: fix scattered punctuation (Georgian-specific)
          .replace(/^[\s\-\.,!?]+/gm, '')                 // Remove leading punctuation on lines
          .replace(/[\s\-\.,!?]+$/gm, '')                 // Remove trailing punctuation on lines
          .replace(/\s*[-]+\s*/g, ' ')                    // Clean up scattered dashes
          .replace(/\s*[\.]{2,}/g, '.')                   // Multiple periods to single
          .replace(/\s*[,]{2,}/g, ',')                    // Multiple commas to single
          .replace(/\s*[\.]\s*[,]/g, '.')                 // Period-comma combinations
          .replace(/\s*[,]\s*[\.]/g, '.')                 // Comma-period combinations
          
          // Third pass: fix spacing around punctuation
          .replace(/\s+([\.!?])/g, '$1')                  // Remove space before sentence endings
          .replace(/\s+([,;:])/g, '$1')                   // Remove space before punctuation
          .replace(/([\.!?])([ა-ჰА-Яa-zA-Z])/g, '$1 $2')  // Add space after sentences
          .replace(/([,;:])([ა-ჰА-Яa-zA-Z])/g, '$1 $2')   // Add space after punctuation
          
          // Fourth pass: clean up remaining artifacts
          .replace(/^\s*[\.!?,;:-]+\s*/g, '')             // Remove punctuation-only beginnings
          .replace(/\s*[\.!?,;:-]+\s*$/g, '')             // Remove punctuation-only endings
          .replace(/\s{2,}/g, ' ')                        // Final whitespace normalization
          .trim();

        console.log(`📝 Original text length: ${data.text.length} chars`);
        console.log(`🧹 Cleaned text length: ${cleanedText.length} chars`);
        console.log(`📊 Cleaning removed: ${data.text.length - cleanedText.length} chars (${Math.round((data.text.length - cleanedText.length) / data.text.length * 100)}%)`);
        
        // Log a sample of cleaned text for debugging
        console.log(`📋 Sample cleaned text: "${cleanedText.substring(0, 200)}..."`);
        
        if (cleanedText.length < data.text.length * 0.5) {
          console.warn(`⚠️ Warning: Aggressive cleaning removed >50% of text. Original might be heavily corrupted.`);
        }
        
        console.log(`📝 Extracted ${cleanedText.length} characters from ${file.originalname}`);

        if (cleanedText.length < 50) {
          console.warn(`⚠️ Warning: Very short text extracted from ${file.originalname} (${cleanedText.length} chars)`);
        }

        // Step 3: Create semantic chunks using LangChain
        const textChunks = await createSemanticChunks(cleanedText);
        console.log(textChunks[0], textChunks[1], textChunks[2]); // Log first 3 chunks for debugging

        if (textChunks.length === 0) {
          throw new Error(`No chunks created from ${file.originalname}`);
        }

        // Step 4: Generate embeddings for semantic chunks
        const embeddingVectors = await generateEmbeddingsForChunks(textChunks);

        // Step 5: Upload semantic chunks to Pinecone
        await uploadChunksToPinecone(textChunks, embeddingVectors, userId, file.originalname);

        results.push({
          filename: file.originalname,
          textLength: cleanedText.length,
          chunksCreated: textChunks.length,
          vectorsUploaded: embeddingVectors.length,
          averageChunkSize: Math.round(textChunks.reduce((sum, chunk) => sum + chunk.charCount, 0) / textChunks.length),
          averageWordsPerChunk: Math.round(textChunks.reduce((sum, chunk) => sum + chunk.wordCount, 0) / textChunks.length),
          chunkingMethod: 'semantic_langchain',
          status: 'success'
        });

        console.log(`✅ Successfully processed ${file.originalname}`);

      } catch (fileError) {
        console.error(`❌ Error processing file ${file.originalname}:`, fileError);
        results.push({
          filename: file.originalname,
          status: 'error',
          error: fileError.message
        });
      } finally {
        // Clean up temp file
        try {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
          }
        } catch (cleanupErr) {
          console.error(`⚠️ Warning: Could not clean up temp file ${file.path}:`, cleanupErr);
        }
      }
    }

    const successfulFiles = results.filter(r => r.status === 'success').length;
    const failedFiles = results.filter(r => r.status === 'error').length;

    console.log(`🎉 Processing complete! ${successfulFiles} successful, ${failedFiles} failed`);

    res.status(200).json({
      success: true,
      message: 'PDF files processed with semantic chunking and indexed successfully',
      results: results,
      totalFiles: req.files.length,
      successfulFiles: successfulFiles,
      failedFiles: failedFiles,
      chunkingMethod: 'semantic_langchain',
      chunkingConfig: {
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP,
        separators: SEPARATORS
      }
    });

  } catch (err) {
    console.error('❌ Error in PDF processing pipeline:', err);
    
    // Clean up any remaining temp files
    if (req.files) {
      req.files.forEach(file => {
        try {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
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