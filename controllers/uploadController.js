const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { pipeline } = require('@xenova/transformers');
const {Pinecone} = require('@pinecone-database/pinecone');
const pdfParse = require('pdf-parse');

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
  chunkSize = 500,
  filename = ""
) {
  console.log(`✂️ [${filename}] Creating fixed-size chunks (${chunkSize} characters each)...`);

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

  console.log(`✅ [${filename}] Created ${textChunks.length} text chunks of ${chunkSize} characters each`);
  return textChunks;
}

async function generateEmbeddingsForChunks(textChunks, filename = "") {
  console.log(`🧠 [${filename}] Initializing embedding model '${EMBEDDING_MODEL_NAME}'...`);
  const embeddingPipeline = await pipeline(
    "feature-extraction",
    EMBEDDING_MODEL_NAME
  );
  console.log(`✅ [${filename}] Embedding model loaded successfully`);

  const embeddingVectors = [];
  const totalChunks = textChunks.length;

  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    if (chunkIndex % 10 === 0 || chunkIndex === totalChunks - 1) {
      console.log(`⚡ [${filename}] Processing chunk ${chunkIndex + 1} of ${totalChunks}...`);
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

  console.log(`✅ [${filename}] Generated ${embeddingVectors.length} embedding vectors`);
  return embeddingVectors;
}

async function uploadChunksToPinecone(textChunks, embeddingVectors, userId, filename) {
  console.log(`💾 [${filename}] Preparing to upload ${textChunks.length} vectors to Pinecone...`);
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
  console.log(`📦 [${filename}] Uploading in ${totalBatches} batches of ${BATCH_SIZE}...`);

  for (let batchIndex = 0; batchIndex < vectorsToUpload.length; batchIndex += BATCH_SIZE) {
    const currentBatch = vectorsToUpload.slice(batchIndex, batchIndex + BATCH_SIZE);
    const batchNumber = Math.floor(batchIndex / BATCH_SIZE) + 1;
    
    try {
      await pineconeIndex.upsert(currentBatch);
      console.log(`✅ [${filename}] Uploaded batch ${batchNumber} of ${totalBatches}`);
    } catch (error) {
      console.error(`❌ [${filename}] Failed to upload batch ${batchNumber}:`, error.message);
      throw new Error(`Batch upload failed for ${filename}: ${error.message}`);
    }
  }
  
  console.log(`🎉 [${filename}] Successfully uploaded all ${vectorsToUpload.length} vectors to Pinecone`);
}

// ===== PARALLEL PROCESSING FUNCTION =====

async function processFileInParallel(file, userId) {
  const startTime = Date.now();
  console.log(`🚀 [${file.originalname}] Starting parallel processing...`);
  
  try {
    // Step 1: Parse PDF
    console.log(`📄 [${file.originalname}] Reading PDF file...`);
    const dataBuffer = fs.readFileSync(file.path);
    const data = await pdfParse(dataBuffer);
    console.log(`📝 [${file.originalname}] PDF parsed successfully`);

    // Clean the extracted text
    const cleanedText = data.text.replace(/\s+/g, ' ').trim();
    console.log(`📝 [${file.originalname}] Extracted ${cleanedText.length} characters`);

    // Step 2: Create text chunks
    const textChunks = await createTextChunks(cleanedText, 500, file.originalname);

    // Step 3: Generate embeddings
    console.log(`🧠 [${file.originalname}] Starting embedding generation...`);
    const embeddingVectors = await generateEmbeddingsForChunks(textChunks, file.originalname);

    // Step 4: Upload to Pinecone
    console.log(`📤 [${file.originalname}] Starting upload to Pinecone...`);
    await uploadChunksToPinecone(textChunks, embeddingVectors, userId, file.originalname);

    // Clean up temp file
    fs.unlinkSync(file.path);
    console.log(`🗑️ [${file.originalname}] Temp file cleaned up`);

    const endTime = Date.now();
    const processingTime = ((endTime - startTime) / 1000).toFixed(2);
    console.log(`✅ [${file.originalname}] Completed in ${processingTime}s`);

    return {
      filename: file.originalname,
      textLength: cleanedText.length,
      chunksCreated: textChunks.length,
      vectorsUploaded: embeddingVectors.length,
      processingTimeSeconds: processingTime,
      status: 'success'
    };

  } catch (error) {
    console.error(`❌ [${file.originalname}] Processing failed:`, error.message);
    
    // Clean up temp file on error
    try {
      if (fs.existsSync(file.path)) {
        fs.unlinkSync(file.path);
        console.log(`🗑️ [${file.originalname}] Temp file cleaned up after error`);
      }
    } catch (cleanupErr) {
      console.error(`⚠️ [${file.originalname}] Error cleaning up temp file:`, cleanupErr);
    }

    throw new Error(`Failed to process ${file.originalname}: ${error.message}`);
  }
}

// ===== SEARCH FUNCTION =====




// ===== MAIN CONTROLLER =====

exports.uploadFiles = async (req, res) => {
  const overallStartTime = Date.now();
  console.log('='.repeat(60));
  console.log('🎯 STARTING PARALLEL PDF PROCESSING PIPELINE');
  console.log('='.repeat(60));
  console.log('Headers:', req.headers);
  console.log('Body:', req.body);
  console.log('Files:', req.files);
  console.log(`📊 Total files to process: ${req.files?.length || 0}`);

  try {
    console.log("🚀 Starting PDF processing and RAG pipeline...");
    
    // Step 1: Ensure Pinecone index exists
    await ensurePineconeIndexExists();
    
    const userId = req.user?._id.toString() || "default_user"; 
    console.log(`👤 Processing for user: ${userId}`);

    // Step 2: Process all files in parallel
    console.log(`🔄 Starting parallel processing of ${req.files.length} files...`);
    const processingPromises = req.files.map((file, index) => {
      console.log(`📋 [File ${index + 1}/${req.files.length}] Queued: ${file.originalname}`);
      return processFileInParallel(file, userId);
    });

    // Wait for all files to complete processing
    console.log(`⏳ Waiting for all ${req.files.length} files to complete...`);
    const results = await Promise.all(processingPromises);

    const overallEndTime = Date.now();
    const totalProcessingTime = ((overallEndTime - overallStartTime) / 1000).toFixed(2);

    console.log('='.repeat(60));
    console.log("🎉 ALL FILES PROCESSED SUCCESSFULLY!");
    console.log(`⏱️ Total processing time: ${totalProcessingTime}s`);
    console.log(`📈 Average time per file: ${(totalProcessingTime / req.files.length).toFixed(2)}s`);
    console.log(`📊 Total chunks created: ${results.reduce((sum, r) => sum + r.chunksCreated, 0)}`);
    console.log(`📊 Total vectors uploaded: ${results.reduce((sum, r) => sum + r.vectorsUploaded, 0)}`);
    console.log('='.repeat(60));

    res.status(200).json({
      success: true,
      message: 'PDF files processed and indexed successfully in parallel',
      results: results,
      totalFiles: req.files.length,
      totalProcessingTimeSeconds: totalProcessingTime,
      averageTimePerFile: (totalProcessingTime / req.files.length).toFixed(2),
      totalChunksCreated: results.reduce((sum, r) => sum + r.chunksCreated, 0),
      totalVectorsUploaded: results.reduce((sum, r) => sum + r.vectorsUploaded, 0)
    });

  } catch (err) {
    console.error('❌ Error in parallel processing pipeline:', err);
    
    // Clean up any remaining temp files
    if (req.files) {
      console.log('🧹 Cleaning up remaining temp files...');
      req.files.forEach((file, index) => {
        try {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
            console.log(`🗑️ Cleaned up temp file ${index + 1}: ${file.originalname}`);
          }
        } catch (cleanupErr) {
          console.error(`⚠️ Error cleaning up temp file ${file.originalname}:`, cleanupErr);
        }
      });
    }

    const overallEndTime = Date.now();
    const totalProcessingTime = ((overallEndTime - overallStartTime) / 1000).toFixed(2);
    console.log(`❌ Pipeline failed after ${totalProcessingTime}s`);

    res.status(500).json({ 
      success: false,
      error: 'Failed to process PDF files in parallel',
      message: err.message,
      processingTimeSeconds: totalProcessingTime
    });
  }
};