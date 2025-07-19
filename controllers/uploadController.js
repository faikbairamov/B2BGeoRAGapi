
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { pipeline } = require('@xenova/transformers');
const {Pinecone} = require('@pinecone-database/pinecone');
const pdfParse = require('pdf-parse');

// Configuration Constants
const BATCH_SIZE = 100;

// Pinecone Configuration
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = "georag";
const PINECONE_INDEX_REGION = "us-east-1";

// Model Configuration
const EMBEDDING_MODEL_NAME = "Xenova/bge-m3";
const EMBEDDING_DIMENSION = 1024;
const SIMILARITY_METRIC = "cosine";

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

async function createTextChunks(
  textContent,
  chunkSize = 500
) {
  console.log(`✂️ Creating fixed-size chunks (${chunkSize} characters each)...`);

  const textChunks = [];
  let position = 0;

  while (position < textContent.length) {
    const chunkText = textContent.slice(position, position + chunkSize);

    textChunks.push({
      id: uuidv4(),
      text: chunkText,
      wordCount: chunkText.split(" ").length,
      startPosition: position,
      charCount: chunkText.length,
    });

    position += chunkSize;
  }

  console.log(`✅ Created ${textChunks.length} text chunks of ${chunkSize} characters each`);
  return textChunks;
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
    
    const embeddingOutput = await embeddingPipeline(
      textChunks[chunkIndex].text,
      {
        pooling: "cls",
        normalize: true,
      }
    );
    embeddingVectors.push(embeddingOutput.data);
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
        startPosition: currentChunk.startPosition,
        chunkId: currentChunk.id,
        uploadedAt: new Date().toISOString(),
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




// ===== MAIN CONTROLLER =====

exports.uploadFiles = async (req, res) => {
  try {
    console.log("🚀 Starting PDF processing and RAG pipeline...");
    
    // Step 1: Ensure Pinecone index exists
    await ensurePineconeIndexExists();
    
    const results = [];
    const userId = req.body.userId || "default_user"; // Get userId from request body or default

    // Step 2: Process each uploaded PDF file
    for (const file of req.files) {
      console.log(`📄 Processing file: ${file.originalname}`);
      
      const dataBuffer = fs.readFileSync(file.path);
      const data = await pdfParse(dataBuffer);

      // Clean the extracted text
      const cleanedText = data.text.replace(/\s+/g, ' ').trim();
      console.log(`📝 Extracted ${cleanedText.length} characters from ${file.originalname}`);

      // Step 3: Create text chunks
      const textChunks = await createTextChunks(cleanedText, 500);

      // Step 4: Generate embeddings
      const embeddingVectors = await generateEmbeddingsForChunks(textChunks);

      // Step 5: Upload to Pinecone
      await uploadChunksToPinecone(textChunks, embeddingVectors, userId, file.originalname);

      results.push({
        filename: file.originalname,
        textLength: cleanedText.length,
        chunksCreated: textChunks.length,
        vectorsUploaded: embeddingVectors.length,
        status: 'success'
      });

      // Clean up temp file
      fs.unlinkSync(file.path);
    }

    console.log("🎉 All files processed successfully!");

    res.status(200).json({
      success: true,
      message: 'PDF files processed and indexed successfully',
      results: results,
      totalFiles: req.files.length
    });

  } catch (err) {
    console.error('❌ Error processing PDF files:', err);
    
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
      error: 'Failed to process PDF files',
      message: err.message 
    });
  }
};