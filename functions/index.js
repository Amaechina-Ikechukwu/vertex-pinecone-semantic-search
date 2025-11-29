const { getStorage } = require("firebase-admin/storage");
const { onObjectFinalized } = require("firebase-functions/v2/storage");
const { onRequest } = require("firebase-functions/v2/https");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore } = require("firebase-admin/firestore");
const { Pinecone } = require("@pinecone-database/pinecone");
const path = require("path");
const logger = require("firebase-functions/logger");
const { GoogleAuth } = require("google-auth-library");

// --- CONFIGURATION ---
const GCP_LOCATION = "us-central1";
const VISION_MODEL = "gemini-2.5-flash";
const FIRESTORE_COLLECTION = "imageDescriptions";
const EMBEDDING_MODEL = "multimodalembedding@001";
const EMULATOR_GCS_URI = "gs://policypal-4meu1.firebasestorage.app/spenser-sembrat-s7W2PXuYGcc-unsplash.jpg";
const EMULATOR_MIME_TYPE = "image/jpeg";
const IMAGE_GEN_MODEL = "gemini-2.5-flash-image";


// --- FIREBASE INIT (Lazy) ---
let adminApp;
let db, storage; // 'storage' is declared here

function getDb() {
  if (!adminApp) {
    adminApp = initializeApp();
    db = getFirestore();
    storage = getStorage(); // 👈 **ADD THIS LINE**
  }
  return db;
}
// 👇 THIS IS THE FUNCTION THAT WAS MISSING
function getStorageBucket() {
  if (!adminApp) getDb(); // This will now correctly initialize 'storage'
  return storage.bucket(); 
}
// --- CORS ---
function setCorsHeaders(req, res) {
  const origin = (req && req.get && req.get("origin")) || "*";
  res.set("Access-Control-Allow-Origin", origin);
  res.set("Vary", "Origin");
  res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.set("Access-Control-Allow-Credentials", "true");
  res.set("Access-Control-Max-Age", "3600");
}

// --- UTILITIES ---
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function retryWithBackoff(fn, maxRetries = 5, initialDelay = 1000) {
  let retries = 0, delay = initialDelay;
  while (retries < maxRetries) {
    try {
      return await fn();
    } catch (error) {
      const isQuotaError = error.status === 429 || (error.cause && error.cause.code === 429);
      if (isQuotaError) {
        retries++;
        if (retries >= maxRetries) throw error;
        const jitter = Math.random() * 1000;
        logger.warn(`Quota error (429). Retrying in ${(delay + jitter) / 1000}s...`);
        await sleep(delay + jitter);
        delay *= 2;
      } else {
        throw error;
      }
    }
  }
}

// --- LAZY LOADERS FOR HEAVY CLIENTS ---
let VertexAI, aiplatform, PredictionServiceClient, helpers;
let predictionServiceClient;

// ...
function getClients() {
  if (!VertexAI) {
    VertexAI = require("@google-cloud/vertexai").VertexAI;
  }
  const vertexAI = new VertexAI({
    project: process.env.GCLOUD_PROJECT,
    location: GCP_LOCATION,
  });
  const visionModel = vertexAI.getGenerativeModel({ model: VISION_MODEL });

  // 👇 **ADD THIS BACK**
  const imageGenModel = vertexAI.getGenerativeModel({
    model: IMAGE_GEN_MODEL,
    // This config at initialization is CRITICAL
    generationConfig: { "responseMimeType": "image/png" },
  });

  return { visionModel, imageGenModel }; // 👈 **RETURN BOTH MODELS**
}


function getPredictionServiceClient() {
  if (!aiplatform) {
    aiplatform = require("@google-cloud/aiplatform");
    PredictionServiceClient = aiplatform.v1.PredictionServiceClient;
    helpers = aiplatform.helpers;
  }
  if (!predictionServiceClient) {
    predictionServiceClient = new PredictionServiceClient({
      apiEndpoint: `${GCP_LOCATION}-aiplatform.googleapis.com`,
    });
  }
  return { predictionServiceClient, helpers };
}

// --- MULTIMODAL EMBEDDING ---
async function generateMultimodalEmbedding({ text, imagePart }) {
  const { predictionServiceClient, helpers } = getPredictionServiceClient();
  const endpoint = `projects/${process.env.GCLOUD_PROJECT}/locations/${GCP_LOCATION}/publishers/google/models/${EMBEDDING_MODEL}`;

  const prompt = {};
  if (text) prompt.text = text;
  if (imagePart) prompt.image = { gcsUri: imagePart.fileData.fileUri };
  if (Object.keys(prompt).length === 0) throw new Error("Either text or an imagePart must be provided.");

  const instanceValue = helpers.toValue(prompt);
  const parameter = helpers.toValue({ dimension: 1408 });
  const request = { endpoint, instances: [instanceValue], parameters: parameter };

  const [response] = await retryWithBackoff(async () => {
    return await predictionServiceClient.predict(request);
  });

  const predictions = response.predictions;
  if (!predictions || predictions.length === 0) throw new Error("No predictions returned from API.");

  const predictionObject = helpers.fromValue(predictions[0]);
  const embedding = predictionObject.textEmbedding || predictionObject.imageEmbedding;
  if (!embedding) throw new Error("No textEmbedding or imageEmbedding found.");

  return embedding;
}

// ====================================================================
// 1. describeAndStoreImage (Storage Trigger)
// ====================================================================
exports.describeAndStoreImage = onObjectFinalized(
  {
    timeoutSeconds: 300,
    memory: "1GiB",
    secrets: ["PINECONE_API_KEY"],
  },
  async (event) => {
    const fileBucket = event.data.bucket;
    const filePath = event.data.name;
    const contentType = event.data.contentType;
    const isEmulator = process.env.FUNCTIONS_EMULATOR === "true";

    const effectiveType = isEmulator ? EMULATOR_MIME_TYPE : contentType;
    if (!effectiveType?.startsWith("image/")) return null;
    if (!filePath) return null;

    const docId = path.basename(filePath);
    const gcsUri = isEmulator ? EMULATOR_GCS_URI : `gs://${fileBucket}/${filePath}`;
    const imageUrl = `https://firebasestorage.googleapis.com/v0/b/${fileBucket}/o/${encodeURIComponent(filePath)}?alt=media`;

    try {
      const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
      const index = pinecone.index("image-search");
      const { visionModel } = getClients();

      const request = {
        contents: [
          {
            role: "user",
            parts: [
              { text: "Describe this image. Respond with ONLY JSON: {title, description}." },
              { fileData: { mimeType: effectiveType, fileUri: gcsUri } },
            ],
          },
        ],
      };

      const result = await retryWithBackoff(async () => await visionModel.generateContent(request));
      const responseText = result?.response?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!responseText) throw new Error("No text returned from Vertex AI.");

      let parsed = {};
      try {
        parsed = JSON.parse(responseText.replace(/```json|```/g, "").trim());
      } catch {
        parsed = { title: "Untitled Image", description: responseText };
      }

      const { title, description } = parsed;
      if (!description) throw new Error("Description missing in parsed JSON.");

      const embedding = await generateMultimodalEmbedding({
        imagePart: { fileData: { mimeType: effectiveType, fileUri: gcsUri } },
      });

      await index.upsert([
        {
          id: docId,
          values: embedding,
          metadata: { title, description, imageUrl, uploadedAt: new Date().toISOString() },
        },
      ]);

      const keywords = Array.from(
        new Set(description.toLowerCase().replace(/[.,!?;:]/g, "").split(/\s+/).filter((k) => k.length > 2))
      );

      await getDb().collection(FIRESTORE_COLLECTION).doc(docId).set({
        docId,
        imagePath: filePath,
        gcsUri,
        imageUrl,
        uploadedAt: new Date().toISOString(),
        title,
        description,
        keywords,
      });

      logger.log(`✅ Stored image ${docId} successfully.`);
      return true;
    } catch (error) {
      logger.error("❌ Error describing image:", error);
      return null;
    }
  }
);

// ====================================================================
// 2. searchImages (Keyword Firestore Search)
// ====================================================================
exports.searchImages = onRequest({ timeoutSeconds: 60 }, async (req, res) => {
  setCorsHeaders(req, res);
  if (req.method === "OPTIONS") return res.status(204).send("");

  try {
    const q = req.query.q;
    const topK = parseInt(req.query.limit, 10) || 5;
    const collectionRef = getDb().collection(FIRESTORE_COLLECTION);

    if (q) {
      const searchKeywords = Array.from(
        new Set(q.toLowerCase().replace(/[.,!?;:]/g, "").split(/\s+/).filter((k) => k.length > 2))
      ).slice(0, 10);

      const snapshot = await collectionRef
        .where("keywords", "array-contains-any", searchKeywords)
        .limit(topK * 5)
        .get();

      const matches = snapshot.docs.map((doc) => {
        const data = doc.data();
        const score = searchKeywords.filter((k) => data.keywords.includes(k)).length;
        return { id: data.docId, score, imageUrl: data.imageUrl, title: data.title };
      });

      const results = matches
        .filter((m) => m.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(({ id, imageUrl, title }) => ({ id, imageUrl, title }));

      res.status(200).json({ results });
    } else {
      const snapshot = await collectionRef.orderBy("uploadedAt", "desc").limit(topK).get();
      const results = snapshot.docs.map((doc) => {
        const d = doc.data();
        return { id: d.docId, imageUrl: d.imageUrl, title: d.title };
      });
      res.status(200).json({ results });
    }
  } catch (error) {
    logger.error("Search error:", error);
    res.status(500).json({ error: "Search failed." });
  }
});

// ====================================================================
// 3. semanticSearchImages (Vector Search via Pinecone)
// ====================================================================
exports.semanticSearchImages = onRequest(
  { timeoutSeconds: 60, memory: "1GiB", secrets: ["PINECONE_API_KEY"] },
  async (req, res) => {
    setCorsHeaders(req, res);
    if (req.method === "OPTIONS") return res.status(204).send("");

    try {
      const textQuery = req.query.q;
      const topK = parseInt(req.query.limit, 10) || 5;
      if (!textQuery) return res.status(400).json({ error: "Missing search query" });

      const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
      const index = pinecone.index("image-search");

      logger.log(`Generating multimodal embedding for: "${textQuery}"`);
      const embedding = await generateMultimodalEmbedding({ text: textQuery });

      const queryResponse = await index.query({
        vector: embedding,
        topK,
        includeMetadata: true,
      });

      const SCORE_THRESHOLD = 0.05;
      const results = queryResponse.matches
        .filter((m) => m.score > SCORE_THRESHOLD)
        .map((m) => ({
          id: m.id,
          score: m.score,
          title: m.metadata.title,
          imageUrl: m.metadata.imageUrl,
        }));

      res.status(200).json({ results });
    } catch (error) {
      logger.error("Semantic search error:", error);
      res.status(500).json({ error: "Semantic search failed" });
    }
  }
);

// ❗️ You must update these placeholders
const REGION = "us-central1"; // Or your specific region
const PROJECT_ID = "policypal-4meu1"; // Your Google Cloud Project ID
const MODEL_ID = "imagen-4.0-ultra-generate-001"; // Or "gemini-1.5-pro-001", etc.

// Construct the HTTPS endpoint URL
const vertexApiUrl = `https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/publishers/google/models/${MODEL_ID}:generateContent`;

// Initialize Google Auth client
const auth = new GoogleAuth({
  scopes: "https://www.googleapis.com/auth/cloud-platform",
});

// Mocking helper from your original code
// const setCorsHeaders = (req, res) => { /* ... */ };

exports.generateImage = onRequest(
  {
    timeoutSeconds: 120,
    memory: "1GiB",
  },
  async (req, res) => {
    // setCorsHeaders(req, res); // Assuming you have this helper
    if (req.method === "OPTIONS") return res.status(204).send("");

    try {
      const { prompt, count = 1 } = req.body;
      const numImages = parseInt(count, 10) || 1;

      if (!prompt) {
        return res.status(400).json({ error: "Missing 'prompt' in request body" });
      }

      logger.log(`Generating ${numImages} image(s) for prompt: "${prompt}"`);

      // 1. Build the request body (matches the SDK's structure)
      const requestBody = {
        contents: [
          {
            role: "user",
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          responseMimeType: "image/png", // REQUIRED for image output
          candidateCount: numImages,
        },
      };

      // 2. Get an access token for the API call
      const client = await auth.getClient();
      const accessToken = (await client.getAccessToken()).token;

      // 3. Call the Vertex AI HTTPS endpoint using fetch
      // (Replaces the imageGenModel.generateContent() call)
      const apiResponse = await fetch(vertexApiUrl, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${accessToken}`,
          "Content-Type": "application/json; charset=utf-8",
        },
        body: JSON.stringify(requestBody),
      });

      if (!apiResponse.ok) {
        const errorBody = await apiResponse.text();
        throw new Error(`API request failed with status ${apiResponse.status}: ${errorBody}`);
      }

      // The API response body is the "result"
      const result = await apiResponse.json();

      // 4. Process the response (this logic is from your original code)
      const response = result; // The top-level object is the response
      if (!response?.candidates || response.candidates.length === 0) {
        throw new Error("No image candidates were generated by the API.");
      }

      // Get the storage bucket (assuming admin is initialized)
      const bucket = getStorage().bucket(); // Replaced getStorageBucket()
      const savedImages = [];
      const slug = prompt.toLowerCase().replace(/[^a-z0-9]/g, "-").slice(0, 30);
      let index = 0;

      for (const candidate of response.candidates) {
        if (candidate.finishReason !== "SAFETY" && candidate.content?.parts?.[0]) {
          const part = candidate.content.parts[0];

          if (part.inlineData?.data) {
            const imageBytesBase64 = part.inlineData.data;
            const imageBuffer = Buffer.from(imageBytesBase64, "base64");

            const fileName = `generated/${Date.now()}-${slug}-${index++}.png`;
            const file = bucket.file(fileName);

            await file.save(imageBuffer, {
              metadata: {
                contentType: "image/png",
                metadata: { generatedFromPrompt: prompt },
              },
            });

            const publicUrl = `https://firebasestorage.googleapis.com/v0/b/${bucket.name}/o/${encodeURIComponent(fileName)}?alt=media`;
            savedImages.push(publicUrl);
            logger.log(`✅ Image saved to: ${publicUrl}`);

          } else if (part.text) {
            logger.warn(`API returned text instead of image: ${part.text}`);
            throw new Error(`Image generation failed. API responded with: "${part.text}"`);
          }
        } else {
          logger.warn(`Image candidate blocked due to: ${candidate.finishReason}`);
        }
      }

      if (savedImages.length === 0) {
        throw new Error("No images were successfully generated. Check logs for safety warnings.");
      }

      res.status(200).json({
        prompt,
        count: savedImages.length,
        imageUrls: savedImages,
      });

    } catch (error) {
      logger.error("❌ Error generating image:", error);
      res.status(500).json({ error: error.message || "Image generation failed." });
    }
  }
);