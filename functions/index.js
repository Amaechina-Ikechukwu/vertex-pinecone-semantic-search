/**
 * 🔥 Firebase AI-Powered Image Indexing + Search with VertexAI + Pinecone
 * - describeAndStoreImage: triggered on upload, describes image, stores to Firestore + Pinecone
 * - searchImages: keyword-based search via Firestore
 * - semanticSearchImages: vector-based search via Pinecone + multimodal embedding
 */

const { onObjectFinalized } = require("firebase-functions/v2/storage");
const { onRequest } = require("firebase-functions/v2/https");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore } = require("firebase-admin/firestore");
const { Pinecone } = require("@pinecone-database/pinecone");
const path = require("path");
const logger = require("firebase-functions/logger");

// --- CONFIGURATION ---
const GCP_LOCATION = "us-central1";
const VISION_MODEL = "gemini-2.5-flash";
const FIRESTORE_COLLECTION = "imageDescriptions";
const EMBEDDING_MODEL = "multimodalembedding@001";
const EMULATOR_GCS_URI = "gs://policypal-4meu1.firebasestorage.app/spenser-sembrat-s7W2PXuYGcc-unsplash.jpg";
const EMULATOR_MIME_TYPE = "image/jpeg";

// --- FIREBASE INIT (Lazy) ---
let adminApp;
let db;
function getDb() {
  if (!adminApp) {
    adminApp = initializeApp();
    db = getFirestore();
  }
  return db;
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

function getClients() {
  if (!VertexAI) {
    VertexAI = require("@google-cloud/vertexai").VertexAI;
  }
  const vertexAI = new VertexAI({
    project: process.env.GCLOUD_PROJECT,
    location: GCP_LOCATION,
  });
  const visionModel = vertexAI.getGenerativeModel({ model: VISION_MODEL });
  return { visionModel };
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
