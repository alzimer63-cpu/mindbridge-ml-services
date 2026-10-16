import os
import pickle
import warnings
import jwt
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Auth (shared JWT secret with the Node/Express server AND the mri-api) ────
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set. Refusing to start.")


def require_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ─── Load XGBoost Model ───────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "alzheimer_xgboost_model.pkl"), "rb") as f:
    xgb_artifacts = pickle.load(f)

xgb_model     = xgb_artifacts["model"]
xgb_scaler    = xgb_artifacts["scaler"]
feature_names = xgb_artifacts["feature_names"]
threshold     = xgb_artifacts["best_threshold"]

# ─── Load reference dataset (for "fetch by record number" autofill) ──────────
_PATIENT_DF = pd.read_csv(os.path.join(BASE_DIR, "alzheimers_disease_data.csv"))

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="MindBridge Clinical-Data API", version="1.0.0")

_cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:5000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Input Schema (text/clinical features) ────────────────────────────────────
class PatientInput(BaseModel):
    Age: int
    Gender: int
    Ethnicity: int
    EducationLevel: int
    BMI: float
    Smoking: int
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryAlzheimers: int
    CardiovascularDisease: int
    Diabetes: int
    Depression: int
    HeadInjury: int
    Hypertension: int
    SystolicBP: int
    DiastolicBP: int
    CholesterolTotal: float
    CholesterolLDL: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: int
    BehavioralProblems: int
    ADL: float
    Confusion: int
    Disorientation: int
    PersonalityChanges: int
    DifficultyCompletingTasks: int
    Forgetfulness: int


# ─── Text/Clinical Prediction Endpoint ────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientInput, _user: dict = Depends(require_auth)):
    try:
        data = patient.model_dump()
        df = pd.DataFrame([data])

        df["CognitiveScore"] = (
            df["MMSE"] + df["FunctionalAssessment"] + df["ADL"]
        ) / 3

        behavioral_cols = [
            "MemoryComplaints", "BehavioralProblems", "Confusion",
            "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
            "Forgetfulness",
        ]
        df["BehavioralSymptomCount"] = df[behavioral_cols].sum(axis=1)

        df["CardioRiskScore"] = (
            df["CardiovascularDisease"] + df["Diabetes"] +
            df["Hypertension"] + df["Smoking"]
        )

        df["CholesterolRatio"] = df["CholesterolLDL"] / (df["CholesterolHDL"] + 1e-5)

        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[59, 65, 70, 75, 80, 91],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        df = df[feature_names]
        X_scaled = xgb_scaler.transform(df)
        prob = float(xgb_model.predict_proba(X_scaled)[0][1])
        prediction = int(prob >= threshold)

        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "diagnosis": "Alzheimer's Positive" if prediction == 1 else "No Alzheimer's Detected",
            "threshold_used": threshold,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Dataset Record Lookup Endpoint ───────────────────────────────────────────
@app.get("/patient-record/{record_number}")
def get_patient_record(record_number: int, _user: dict = Depends(require_auth)):
    """Look up a clinical record from the reference dataset by its PatientID,
    or (if no PatientID matches) by its 1-indexed row position in the CSV."""
    match = _PATIENT_DF[_PATIENT_DF["PatientID"] == record_number]
    if not match.empty:
        row = match.iloc[0]
    elif 1 <= record_number <= len(_PATIENT_DF):
        row = _PATIENT_DF.iloc[record_number - 1]
    else:
        min_id, max_id = int(_PATIENT_DF["PatientID"].min()), int(_PATIENT_DF["PatientID"].max())
        raise HTTPException(
            status_code=404,
            detail=f"No record found for {record_number}. Use a PatientID ({min_id}-{max_id}) "
                   f"or a row number (1-{len(_PATIENT_DF)}).",
        )

    data = {}
    for field in PatientInput.model_fields:
        value = row[field]
        data[field] = value.item() if hasattr(value, "item") else value

    return {
        "patientId": int(row["PatientID"]),
        "diagnosis": int(row["Diagnosis"]),
        "data": data,
    }


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": {
            "type": "XGBoost (Tuned)",
            "features": len(feature_names),
            "threshold": threshold,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )
