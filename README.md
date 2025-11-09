

# AI Image Search Backend (Firebase + Vertex AI + Pinecone)

This repository contains the Firebase Cloud Functions backend for a smart, AI-powered image search application.

It automatically analyzes uploaded images using Google's **Vertex AI (Gemini)** to generate descriptions and **multimodal embeddings**. It stores text metadata in **Firestore** (for keyword search) and vector embeddings in **Pinecone** (for semantic vector search), enabling two complementary search modes.

# Features

- Automatic AI analysis: the Cloud Function triggers on image upload to your Google Cloud Storage bucket.
- Rich metadata: the Gemini model (`gemini-2.5-flash`) produces a `title` and detailed `description` for each image.
- Vector embeddings: images are converted to 1408-dimensional embeddings using Vertex AI `multimodalembedding@001`.
- Dual search system:
    - Keyword search: fast, Firestore-based endpoint (`/searchImages`) that matches text against stored keywords/metadata.
    - Semantic search: Pinecone vector search (`/semanticSearchImages`) that finds contextually similar images for natural-language queries.
- Robust & scalable: implemented with Firebase Functions (v2). Includes retry logic for transient API/quota errors.

# How it works

1. User uploads an image (e.g., `my-photo.jpg`) to the configured Google Cloud Storage bucket.
2. The `describeAndStoreImage` Cloud Function triggers on the upload event.
3. The function calls Vertex AI (Gemini) to produce a `title` and `description` for the image.
4. The function requests a multimodal embedding from Vertex AI (`multimodalembedding@001`).
5. Text metadata (title, description, keywords) is written to Firestore.
6. The image's vector embedding and metadata are upserted into the Pinecone index.
7. Frontend clients can query either the keyword or semantic search endpoints to find images.

# API endpoints

Both endpoints are simple `GET` requests and accept URL query parameters.

1) Keyword search (Firestore)

- Purpose: perform a keyword/text match against titles/descriptions/keywords stored in Firestore.
- Endpoint: `.../searchImages`
- Query params:
    - `q` (string, optional) — search query. If omitted, returns the most recent images.
    - `limit` (number, optional) — number of results to return (default: 5).
- Example request:

    GET https://<your-function-url>/searchImages?q=dog%20on%20a%20leash&limit=10

2) Semantic search (Pinecone)

- Purpose: perform a vector similarity search using Pinecone to find contextually similar images.
- Endpoint: `.../semanticSearchImages`
- Query params:
    - `q` (string, required) — natural language query.
    - `limit` (number, optional) — number of results to return (default: 5).
- Example request:

    GET https://<your-function-url>/semanticSearchImages?q=a%20peaceful%20day%20outdoors&limit=3

Example semantic response (schema):

```
[
    {
        "id": "img_123",
        "score": 0.892,
        "metadata": {
            "title": "Sunny beach",
            "description": "A beach scene with sun and umbrellas",
            "firestorePath": "images/img_123"
        }
    }
]
```

## 🛠️ Setup and deployment

### 1) Prerequisites

- A Firebase project (Blaze plan recommended for v2 functions and external network access).
- A Pinecone account and an index. NOTE: your Pinecone index must have dimension **1408** to match Vertex AI embeddings.
- Google Cloud APIs enabled: Vertex AI, Cloud Functions, Cloud Build, Cloud Storage.

Also ensure your Firebase Functions runtime and Node version are compatible (project currently uses Firebase Functions v2; Node 18+ is recommended). Check `functions/package.json` for exact dependency constraints.

### 2) Local setup

1. Clone the repository:

```sh
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies:

```sh
npm install
```

3. Initialize Firebase functions if you haven't already:

```sh
firebase init functions
```

4. (Optional) Use the Firebase emulators to test locally (Firestore, Functions, Storage):

```sh
firebase emulators:start --only functions,firestore,storage
```

### 3) Configuration (required)

The function needs a Pinecone API key and other configuration values. Recommended variables:

- `PINECONE_API_KEY` (secret) — Pinecone API key.
- `PINECONE_INDEX` — Pinecone index name.
- `PINECONE_ENV` or `PINECONE_REGION` — Pinecone environment/region (if applicable).
- `GCP_PROJECT` — optional override of the Google Cloud project id.

Set the Pinecone secret in Firebase:

```sh
firebase functions:secrets:set PINECONE_API_KEY
```

For other config values you can use Firebase environment config or store small values in `functions/config` as needed.

### 4) Deployment

Deploy functions to Firebase:

```sh
firebase deploy --only functions
```

Note: the Blaze billing plan is required for outbound network calls to external APIs (Pinecone, Vertex) when running on Cloud Functions v2.

### 5) Pinecone index

Create a Pinecone index (via console or CLI) with dimension 1408. See Pinecone documentation for example CLI or dashboard steps.

Example (informational only):

```text
# In Pinecone console or using their CLI, create an index with dimension=1408
```
## Troubleshooting & logs

- Check Cloud Functions logs in the Firebase console or via:

```sh
firebase functions:log
```
-
### Local curl testing examples

When running the Functions emulator locally you can test the semantic search endpoint with curl. Example requests and responses from a local run:

```
curl "http://127.0.0.1:5001/policypal-4meu1/us-central1/semanticSearchImages?q=rat&limit=3"
{"results":[]}

curl "http://127.0.0.1:5001/policypal-4meu1/us-central1/semanticSearchImages?q=horse&limit=3"
{"results":[{"id":"laura-cleffmann-gRT7o73xua0-unsplash.jpg","score":0.0757369921,"title":"Majestic Eagle Hunter on Horseback Amidst Snowy Peaks","imageUrl":"https://firebasestorage.googleapis.com/v0/b/policypal-4meu1.firebasestorage.app/o/laura-cleffmann-gRT7o73xua0-unsplash.jpg?alt=media"}]}

curl "http://127.0.0.1:5001/policypal-4meu1/us-central1/semanticSearchImages?q=a+dog+on+a+beach&limit=3"
{"results":[]}
```
- Common issues:
    - Authentication/permissions: ensure the Functions service account has access to Vertex AI and Cloud Storage.
    - Pinecone auth errors: confirm `PINECONE_API_KEY` is set and the index name matches.
    - Quota errors: Vertex AI or Cloud Build quota limits may require retries or quota increases.

## Links and references

- [Vertex AI — Generative AI quickstart (Node.js gen‑AI SDK)](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=adc#node.js-gen-ai-sdk) — official quickstart and SDK docs for Node.js.
- [Pinecone — Get started overview](https://docs.pinecone.io/guides/get-started/overview) — Pinecone quickstart and index setup guide.

## Next steps / optional improvements

- Add example cURL or Postman collections for the endpoints.
- Add unit/integration tests for the functions (emulator-backed).
- Provide a small sample frontend demonstrating how to upload and query images.

---

