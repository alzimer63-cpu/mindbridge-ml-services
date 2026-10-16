# MindBridge ML services (split)

Two independently deployable FastAPI services, split out of what used to be
one combined `alzimer_backend` service, specifically so each one's dependency
footprint (and therefore memory use and deployment-size budget) is as small
as possible — and so `xgboost`'s unpredictable wheel size (4MB or 147MB,
platform-dependent) no longer affects whether the CNN service fits a given
host's limits.

| Service | Serves | Key dependencies | Does NOT need |
|---|---|---|---|
| `clinical-api/` | `/predict`, `/patient-record/{id}` | xgboost, scikit-learn, pandas, numpy | onnxruntime, pillow |
| `mri-api/` | `/predict-mri` | onnxruntime, pillow | xgboost, scikit-learn, pandas |

Both still verify JWTs issued by the same Node/Express API (`alzimer-frontend/server.js`
in the main project) — `JWT_SECRET` must be identical across all three
services. Both also expose `/health`.

## Local setup

```bash
# clinical-api
cd clinical-api
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # JWT_SECRET must match the Node API's and mri-api's
python main.py         # :8000

# mri-api (separate terminal)
cd mri-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py         # :8001
```

The main project's `alzimer-frontend/vite.config.js` already proxies
`/predict` + `/patient-record` to `:8000` and `/predict-mri` to `:8001` in
dev, and `src/api/api.js` calls them via two separate axios instances
(`clinicalApi`, `mriApi`) — so local dev now runs **four** processes total:
node-api, clinical-api, mri-api, and the Vite dev server. Verified working
end-to-end through the real frontend's proxy.

## The CNN model file

`mri-api/alzheimer_cnn_model.onnx` (~94MB) isn't included if you clone this
elsewhere without copying it manually — same reasoning as before (GitHub's
25MB web-upload limit, `.gitignore`'d). `mri-api/main.py` downloads it from
`CNN_MODEL_URL` on first boot if it isn't present locally. `clinical-api` has
no equivalent problem — its model file is small enough to commit directly.

## Deploying

Deploy `clinical-api/` and `mri-api/` as two **separate** web services on
whatever host you choose (Render, a VPS, etc. — each is just a plain
`pip install -r requirements.txt` + `python main.py` FastAPI app, no
special packaging needed). Each needs its own `JWT_SECRET` (same value both
places) and `CORS_ORIGINS` (your frontend's URL) set as environment
variables. `mri-api` additionally needs `CNN_MODEL_URL` unless you place the
`.onnx` file directly on that host.

Once both are deployed, set on the frontend (Vercel or wherever it's hosted):
- `VITE_CLINICAL_API_URL` = clinical-api's deployed URL
- `VITE_MRI_API_URL` = mri-api's deployed URL

Both are baked in at build time (Vite), so set them before building/deploying
the frontend, then redeploy if you change them later.
