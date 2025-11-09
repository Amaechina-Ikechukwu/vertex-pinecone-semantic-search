const { onObjectFinalized } = require("firebase-functions/v2/storage");
const { onRequest } = require("firebase-functions/v2/https");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore } = require("firebase-admin/firestore");
const { VertexAI } = require("@google-cloud/vertexai");
const logger = require("firebase-functions/logger");
const path = require("path");
const { Pinecone } = require("@pinecone-database/pinecone");
const aiplatform = require("@google-cloud/aiplatform");
const { PredictionServiceClient } = aiplatform.v1;
const { helpers } = aiplatform;

/**
 * Helper to consistently set CORS headers on HTTP responses.
 * Uses the request Origin when present and sets Vary: Origin to be correct for caching.
 */
function setCorsHeaders(req, res) {
  const origin = (req && req.get && req.get('origin')) || '*';
  res.set('Access-Control-Allow-Origin', origin);
  // Signal caches that responses vary by Origin header
  res.set('Vary', 'Origin');
  res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.set('Access-Control-Allow-Credentials', 'true');
  res.set('Access-Control-Max-Age', '3600');
}

// --- FIX 1: LAZY INITIALIZATION for Firebase Admin ---
// We remove initializeApp() and getFirestore() from the global scope.
let adminApp;
let db;

/**
 * Lazily initializes and returns the Firestore database instance.
 * This prevents the 10-second deployment timeout.
 */
function getDb() {
  if (!adminApp) {
    adminApp = initializeApp(); // Initialize only if it hasn't been.
    db = getFirestore();
  }
  return db;
}
// ----------------------------------------------------

// --- Configuration ---
const GCP_LOCATION = "us-central1";
const VISION_MODEL = "gemini-2.5-flash";
const FIRESTORE_COLLECTION = "imageDescriptions";

// --- NEW: Hardcoded URL for Emulator ---
const EMULATOR_GCS_URI = "gs://policypal-4meu1.firebasestorage.app/spenser-sembrat-s7W2PXuYGcc-unsplash.jpg";
const EMULATOR_MIME_TYPE = "image/jpeg";
// ----------------------------------------

// --- Retry Helper Functions (Still very useful!) ---

/**
 * A simple promise-based sleep function.
 * @param {number} ms Time to sleep in milliseconds.
 */
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retries a function with exponential backoff, specifically for 429 quota errors.
 * @param {Function} fn The async function to retry.
 * @param {number} maxRetries Maximum number of retries.
 * @param {number} initialDelay Starting delay in milliseconds.
 */
async function retryWithBackoff(fn, maxRetries = 5, initialDelay = 1000) {
  let retries = 0;
  let delay = initialDelay;

  while (retries < maxRetries) {
    try {
      return await fn(); // Attempt the function
    } catch (error) {
      // Check if it's a quota error (429)
      const isQuotaError = error.status === 429 || (error.cause && error.cause.code === 429);
      
      if (isQuotaError) {
        retries++;
        if (retries >= maxRetries) {
          logger.error("Max retries reached for quota error.", { error: error.message });
          throw error; // Rethrow after max retries
        }
        
        const jitter = Math.random() * 1000;
        logger.warn(`Quota error (429). Retrying in ${Math.round((delay + jitter)/1000)}s... (Attempt ${retries}/${maxRetries})`);
        await sleep(delay + jitter);
        delay *= 2; // Exponential backoff
      } else {
        // Not a quota error, throw immediately
        logger.error("Non-quota error encountered:", error);
        throw error;
      }
    }
  }
}

// --- End of new helpers ---


/**
 * Helper function to initialize clients
 */
const getClients = () => {
  const vertexAI = new VertexAI({
    project: process.env.GCLOUD_PROJECT,
    location: GCP_LOCATION,
  });

  const visionModel = vertexAI.getGenerativeModel({
    model: VISION_MODEL,
  });

  return { visionModel };
};

/**
 * =========================================================================
 * 1. INDEXING FUNCTION (Now Describe & Store)
 * Triggers when a new image is uploaded to Storage.
 * =========================================================================
 */
exports.describeAndStoreImage = onObjectFinalized(
  {
    timeoutSeconds: 300,
    memory: "1GiB",
    // --- FIX 1: Tell this function to access the secret ---
    secrets: ["PINECONE_API_KEY"],
  },
  async (event) => {
    // --- Get event data ---
    const fileBucket = event.data.bucket;
    const filePath = event.data.name;
    const contentType = event.data.contentType;

    // --- NEW: Check if running in emulator ---
    const isEmulator = process.env.FUNCTIONS_EMULATOR === "true";

    // 1. Validate file (is it an image?)
    const effectiveContentType = isEmulator ? EMULATOR_MIME_TYPE : contentType;
    if (!effectiveContentType || !effectiveContentType.startsWith("image/")) {
      logger.log("This is not an image.");
      return null;
    }
    if (!filePath) {
      logger.log("File path is undefined.");
      return null;
    }

    const docId = path.basename(filePath);
    logger.log(`Processing image: ${docId}`);

    // --- UPDATED: Determine which GCS URI to send to Vertex AI ---
    let gcsUriForVertex;
    if (isEmulator) {
      gcsUriForVertex = EMULATOR_GCS_URI;
      logger.warn(`--- EMULATOR MODE ---`);
      logger.warn(`Triggered by: gs://${fileBucket}/${filePath}`);
      logger.warn(`Processing (hardcoded): ${gcsUriForVertex}`);
    } else {
      gcsUriForVertex = `gs://${fileBucket}/${filePath}`;
    }
    // -------------------------------------------------------------

    try {
      // --- FIX 2: Initialize Pinecone *inside* the function ---
      // process.env.PINECONE_API_KEY is now available thanks to the 'secrets' option
      const pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });
      const index = pinecone.index("image-search");
      // -----------------------------------------------------------

      const { visionModel } = getClients();

      // 2. Prepare the description request for Vertex AI
      const imagePart = {
        fileData: { 
          mimeType: effectiveContentType, 
          fileUri: gcsUriForVertex 
        },
      };
      const textPart = {
        text: "Describe this image. Respond with ONLY a valid JSON object containing two keys: 'title' (a short, catchy title for the image, 5-10 words) and 'description' (a detailed description focusing on objects, setting, colors, and any text present for a search index)."
      };
      
      const request = {
        contents: [
          {
            role: "user",
            parts: [textPart, imagePart]
          }
        ],
      };

      // 3. Generate the description
      logger.log("Generating description...");
      
      const result = await retryWithBackoff(async () => {
        return await visionModel.generateContent(request);
      });
      
      const response = result.response;

      // 4. Parse the JSON response from Vertex AI
      const responseText = response?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!responseText) {
        logger.error("Unexpected response structure from Vertex AI. No text part found.", { response: JSON.stringify(response) });
        throw new Error("No text returned from Vertex AI.");
      }

      let parsedResponse = {};
      try {
        // The model should return a raw JSON string. We clean it up just in case.
        const jsonString = responseText.replace(/```json\n?/g, "").replace(/\n?```/g, "").trim();
        parsedResponse = JSON.parse(jsonString);
      } catch (e) {
        logger.error("Failed to parse JSON from Vertex AI response, using fallback.", { responseText, error: e.message });
        // Fallback: use the whole text as description if parsing fails
        parsedResponse = { title: "Untitled Image", description: responseText };
      }

      const { title, description } = parsedResponse;
      
      if (!description) {
           logger.error("Description not found in parsed JSON response.", { parsedResponse });
           throw new Error("No description in parsed JSON from Vertex AI.");
      }
      
      logger.log(`Description generated: ${description.substring(0, 60)}...`);

      // --- FIX 3: Define imageUrl *before* it's needed by Pinecone ---
      const encodedFilePath = encodeURIComponent(filePath);
      const imageUrl = `https://firebasestorage.googleapis.com/v0/b/${fileBucket}/o/${encodedFilePath}?alt=media`;
      // -------------------------------------------------------------

      // --- NEW: Generate vector embedding for the description ---
      logger.log("Generating multimodal vector embedding for the image...");
      // We pass the same imagePart we prepared for Gemini earlier
      const embedding = await generateMultimodalEmbedding({ imagePart: imagePart });

      // --- Store in Pinecone ---
      logger.log("Upserting to Pinecone...");
      await index.upsert([
        {
          id: docId,
          values: embedding, // <-- This is now the IMAGE vector
          metadata: {
            title,
            description, // <-- We still store the text description as metadata
            imageUrl,
            uploadedAt: new Date().toISOString(),
          },
        },
      ]);
      logger.log(`Vector stored in Pinecone for ${docId}.`);

      // 5. Create keywords for searching
      const keywords = Array.from(new Set( // Use a Set to remove duplicates
        description.toLowerCase()
          .replace(/[.,!?;:]/g, '') // Remove simple punctuation
          .split(/\s+/) // Split on whitespace
          .filter(k => k.length > 2) // Filter out short words (like 'a', 'is', 'to')
      ));

      // 6. Save to Firestore (replaces Pinecone upsert)
      logger.log("Saving description to Firestore...");
      
      // --- FIX 4: Use the getDb() helper function ---
      const docRef = getDb().collection(FIRESTORE_COLLECTION).doc(docId);
      
      await docRef.set({
        docId: docId,
        imagePath: filePath,
        gcsUri: `gs://${fileBucket}/${filePath}`,
        imageUrl: imageUrl,
        uploadedAt: new Date().toISOString(),
        title: title,
        description: description,
        keywords: keywords
       });

      logger.log(`Successfully described and stored image ${docId}.`);

      return true;

    } catch (error) {
      logger.error("Error describing image:", error.message, {fullError: error});
      return null;
    }
  }
);

/**
 * =========================================================================
 * 2. SEARCH FUNCTION (Now Firestore Keyword Search)
 * An HTTP function your app will use to search for images.
 * =========================================================================
 */
// This function doesn't use Pinecone, so no changes needed
exports.searchImages = onRequest(
  {
    timeoutSeconds: 60,
  },
  async (req, res) => { // Using (req, res) signature
    // --- CORS Handling ---
    // Use the helper so the Access-Control headers are always present
    setCorsHeaders(req, res);

    if (req.method === "OPTIONS") {
      // Preflight: headers already set by setCorsHeaders
      res.status(204).send("");
      return;
    }
    // --------------------------

    try {
      // --- REWRITTEN: Get params from req.query (URL) instead of req.body ---
      // e.g., /searchImages?q=dog&limit=10
      const textQuery = req.query.q; // 'q' is standard for query
      // Parse 'limit' from query, default to 5. Ensure it's a number.
      const topK = parseInt(req.query.limit, 10) || 5; 
      // -------------------------------------------------------------------

      // --- FIX 4: Use the getDb() helper function ---
      const collectionRef = getDb().collection(FIRESTORE_COLLECTION);

      if (textQuery) {
        // --- A. HAS A SEARCH QUERY: Perform keyword search ---
        logger.log(`Performing search for: "${textQuery}"`);

        // 1. Generate search keywords from the text query
        const searchKeywords = Array.from(new Set(
          textQuery.toLowerCase()
            .replace(/[.,!?;:]/g, '')
            .split(/\s+/)
            .filter(k => k.length > 2)
       ));
        
        if (searchKeywords.length === 0) {
          logger.log("No valid search keywords from query.");
          res.status(200).json({ results: [] });
          return;
        }

        // Firestore 'array-contains-any' is limited to 10 values in the query
        const queryKeywords = searchKeywords.slice(0, 10);
        if (searchKeywords.length > 10) {
          logger.warn(`Query truncated to 10 keywords for Firestore limit: ${queryKeywords.join(', ')}`);
        }
        logger.log(`Searching for keywords: ${queryKeywords.join(', ')}`);

        // 2. Query Firestore
        const querySnapshot = await collectionRef
          .where('keywords', 'array-contains-any', queryKeywords)
          .limit(topK * 5) // Get extra results to rank in-memory
          .get();

        if (querySnapshot.empty) {
          logger.log("No documents found with 'array-contains-any'.");
          res.status(200).json({ results: [] });
          return;
        }

        // 3. Rank results in-memory
        const matches = querySnapshot.docs.map(doc => {
          const data = doc.data();
          let score = 0;
          const docKeywords = new Set(data.keywords);
          for (const keyword of searchKeywords) {
            if (docKeywords.has(keyword)) {
              score++;
            }
          }
          return {
            id: data.docId,
            score: score,
            imageUrl: data.imageUrl,
            title: data.title
          };
        });

        // Filter out 0-score matches, sort by score, and then map to the final structure
        const rankedMatches = matches
          .filter(match => match.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, topK)
          .map(({ id, imageUrl, title }) => ({ id, imageUrl, title })); // Keep only id, imageUrl, title

        res.status(200).json({ results: rankedMatches });

      } else {
        // --- B. NO SEARCH QUERY: Return recent images ---
        logger.log(`No text query. Fetching ${topK} most recent images.`);

        // Use 'topK' (which got 'limit' or 5)
        const querySnapshot = await collectionRef
          .orderBy("uploadedAt", "desc") // Get the newest first
          .limit(topK) // Use the same 'limit' param
          .get();

        if (querySnapshot.empty) {
          res.status(200).json({ results: [] });
          return;
        }

        // Map to the desired structure
        const recentImages = querySnapshot.docs.map(doc => {
          const data = doc.data();
          return {
            id: data.docId,
            imageUrl: data.imageUrl,
             title: data.title
          };
        });

        res.status(200).json({ results: recentImages });
      }

    } catch (error) {
      logger.error("Error searching images:", error.message, {fullError: error});
  // Send a 500 Internal Server Error response
      res.status(500).json({ error: "Search failed due to an internal error." });
    }
  }
);


// --- NEW: Multimodal Embedding Configuration ---
const EMBEDDING_MODEL = "multimodalembedding@001"; // The model from the docs

// --- Client options for PredictionServiceClient ---
// We must specify the regional endpoint
const clientOptions = {
  // --- FIX 2: Corrected typo from GCS_LOCATION to GCP_LOCATION ---
  apiEndpoint: `${GCP_LOCATION}-aiplatform.googleapis.com`,
};

// --- Instantiate the new client ---
// We can do this once here, as it's safe.
// --- Declare the client, but DO NOT initialize it ---
// We will initialize it on the first function call.
let predictionServiceClient;


/**
 * Generates vector embeddings using the multimodal embedding model
 * via the PredictionServiceClient.
 *
 * @param {object} params
 * @param {string} [params.text] - Optional text to embed.
 * @param {object} [params.imagePart] - Optional image part to embed (e.g., { fileData: ... }).
 * @returns {Promise<number[]>} The embedding vector.
 */
async function generateMultimodalEmbedding({ text, imagePart }) {
       if (!predictionServiceClient) {
    predictionServiceClient = new PredictionServiceClient(clientOptions);
  }
  // 1. Configure the parent resource
  const endpoint = `projects/${process.env.GCLOUD_PROJECT}/locations/${GCP_LOCATION}/publishers/google/models/${EMBEDDING_MODEL}`;

  // 2. Build the "instance" (the data to send)
  const prompt = {};
  if (text) {
    prompt.text = text;
  }
  if (imagePart) {
    // The API expects { "image": { "gcsUri": "gs://..." } }
    // Our imagePart is { fileData: { mimeType: ..., fileUri: "gs://..." } }
    prompt.image = { gcsUri: imagePart.fileData.fileUri };
  }

  if (Object.keys(prompt).length === 0) {
    throw new Error("Either text or an imagePart must be provided.");
  }

  // Convert the prompt object into the special "Value" format
  const instanceValue = helpers.toValue(prompt);
  const instances = [instanceValue];

  // 3. Build the parameters (e.g., embedding dimension)
  const embeddingDimension = 1408; // Valid: 128, 256, 512, 1408
  const parameter = {
    dimension: embeddingDimension,
  };
  const parameters = helpers.toValue(parameter);

  // 4. Build the final request
  const request = {
    endpoint,
    instances,
    parameters,
  };

  try {
    // 5. Call the predict method (wrapped in our retry logic)
    const [response] = await retryWithBackoff(async () => {
      return await predictionServiceClient.predict(request);
    });

    // 6. Parse the response
    const predictions = response.predictions;
    if (!predictions || predictions.length === 0) {
      throw new Error("No predictions returned from API.");
    }

    // Convert the "Value" format back into a standard JS object
    const predictionObject = helpers.fromValue(predictions[0]);
    
    // The object will have EITHER "textEmbedding" OR "imageEmbedding"
    const embedding = predictionObject.textEmbedding || predictionObject.imageEmbedding;

    if (!embedding) {

      logger.error("No textEmbedding or imageEmbedding found in prediction", { predictionObject });
      throw new Error("Failed to parse embedding from API response.");
    }

    return embedding; // This is the vector array [0.1, 0.2, ...]

  } catch (error) {
    logger.error("Error in generateMultimodalEmbedding (PredictionServiceClient):", error.message, { fullError: error });
    throw new Error(`Multimodal embedding failed: ${error.message}`);
  }
}





exports.semanticSearchImages = onRequest(
  {
    timeoutSeconds: 60,
    secrets: ["PINECONE_API_KEY"],
    memory: "1GiB"
  },
  async (req, res) => {
    // Ensure CORS headers are present on every response
    setCorsHeaders(req, res);
    if (req.method === "OPTIONS") {
      // Preflight
      res.status(204).send("");
      return;
    }

    try {
      const pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });
      const index = pinecone.index("image-search");

      const textQuery = req.query.q;
      const topK = parseInt(req.query.limit, 10) || 5;

      if (!textQuery) {
        res.status(400).json({ error: "Missing search query" });
        return;
      }

      logger.log(`Generating multimodal embedding for search query: "${textQuery}"`);
    const embedding = await generateMultimodalEmbedding({ text: textQuery });
      
      const queryResponse = await index.query({
        vector: embedding,
        topK,
        includeMetadata: true,
      });

      // --- ADD THIS LINE ---
      // Set a minimum score. You can (and should) adjust this value!
      const SCORE_THRESHOLD = 0.05;

      // --- MODIFY THIS LINE (add the .filter()) ---
      const results = queryResponse.matches
        .filter(match => match.score > SCORE_THRESHOLD) // <-- This filters out bad results
        .map(match => {
          // Log the match id, score, and the threshold for debugging/analysis
          console.log(`match.id=${match.id} match.score=${match.score} SCORE_THRESHOLD=${SCORE_THRESHOLD}`);
          return {
            id: match.id,
            score: match.score,
            title: match.metadata.title,
            imageUrl: match.metadata.imageUrl,
          };
        });
      // ---------------------------------------------

      res.status(200).json({ results });
    } catch (error) {
      logger.error("Error performing semantic search:", error.message, { fullError: error });
      res.status(500).json({ error: "Semantic search failed" });
    }
  }
);