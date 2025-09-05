import streamlit as st
import faiss
import numpy as np
import pickle
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Optional
import re
import time

# ==============================
#   CONFIG
# ==============================
genai.configure(api_key=st.secrets.user.pass1)
MODEL_NAME = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/embedding-001"

INDEX_PATH = "./files/faiss_index.bin"
EMBEDDINGS_PATH = "./files/embeddings.npy"
CHUNKS_PATH = "./files/text_chunks.pkl"

# ==============================
#   DATA MODELS
# ==============================
class Patient(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    performance_status: Optional[str] = None
    comorbidities: Optional[List[str]] = None
    additional_notes: Optional[str] = None   

class Tumor(BaseModel):
    site: str
    histology: str
    stage: str
    molecular_features: Optional[str] = None
    imaging_findings: Optional[str] = None   

class PriorTherapy(BaseModel):
    therapy_type: str
    details: Optional[str] = None

class Recommendation(BaseModel):
    therapy_type: str
    details: str
    level_of_evidence: Optional[str] = None

class AIOutput(BaseModel):
    query: str
    retrieved_context: List[str]
    generated_response: str
    recommendations: List[Recommendation]

# ==============================
#   UTILITIES
# ==============================
@st.cache_resource
def load_artifacts():
    index = faiss.read_index(INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, embeddings, chunks

def build_query_from_structured_input(patient: Patient, tumor: Tumor, prior_therapy: Optional[PriorTherapy] = None) -> str:
    query_parts = []

    # Tumor details
    tumor_info = f"{tumor.histology} of the {tumor.site}, stage {tumor.stage}"
    if tumor.molecular_features:
        tumor_info += f" with {tumor.molecular_features}"
    if tumor.imaging_findings:  # NEW ✅
        tumor_info += f". Imaging findings: {tumor.imaging_findings}"
    query_parts.append(tumor_info)

    # Patient details
    patient_info = []
    if patient.age:
        patient_info.append(f"age {patient.age}")
    if patient.gender:
        patient_info.append(patient.gender)
    if patient.performance_status:
        patient_info.append(f"performance status {patient.performance_status}")
    if patient.comorbidities:
        patient_info.append("comorbidities: " + ", ".join(patient.comorbidities))
    if patient_info:
        query_parts.append("in a " + ", ".join(patient_info))

    # Extra clinical notes
    if patient.additional_notes:  # NEW ✅
        query_parts.append(f"Additional clinical notes: {patient.additional_notes}")

    # Prior therapy, if any
    if prior_therapy:
        prior_text = f"with prior therapy {prior_therapy.therapy_type}"
        if prior_therapy.details:
            prior_text += f" ({prior_therapy.details})"
        query_parts.append(prior_text)

    return (
        "According to the NCCN guidelines, what is the recommended treatment for a "
        + " ".join(query_parts)
        + "?"
    )

def retrieve_chunks(query: str, index, chunks: List[str], top_k: int = 3):
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    distances, indices = index.search(np.array(query_embedding, dtype="float32").reshape(1, -1), top_k)
    return [chunks[i] for i in indices[0]]

def generate_rag_response(query: str, retrieved_chunks: List[str]):
    context = "\n".join(retrieved_chunks)
    prompt = f"Based on the following NCCN guideline excerpts:\n\n{context}\n\nAnswer:\n{query}"
    model = genai.GenerativeModel(MODEL_NAME)
    return model.generate_content(prompt).text

def parse_ai_response(query: str, retrieved: List[str], response_text: str) -> tuple[AIOutput, str]:
    """
    Parses AI output into a structured AIOutput object and also produces a clean, 
    human-readable multiline string for display or logging.
    """

    # -------------------------
    # Structured Recommendation Parsing
    # -------------------------
    recs = []
    rec_pattern = re.compile(r'^[\-\*\+\d\.]\s*(.*)', re.MULTILINE)
    level_pattern = re.compile(r'Level of evidence: (.*)', re.IGNORECASE)

    lines = response_text.strip().splitlines()
    last_idx = -1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        rec_match = rec_pattern.match(line)
        if rec_match:
            details = rec_match.group(1).strip()
            therapy_type = "Unknown"

            for keyword, label in [
                ("surg", "Surgery"),
                ("radio", "Radiotherapy"),
                ("chemo", "Chemotherapy"),
                ("immuno", "Immunotherapy"),
                ("target", "Targeted Therapy"),
                ("systemic", "Systemic Therapy"),
            ]:
                if re.search(keyword, details, re.IGNORECASE):
                    therapy_type = label
                    break

            recs.append(Recommendation(therapy_type=therapy_type, details=details))
            last_idx = len(recs) - 1

        else:
            level_match = level_pattern.search(line)
            if level_match and last_idx >= 0:
                recs[last_idx].level_of_evidence = level_match.group(1).strip()

    # -------------------------
    # Build AIOutput object
    # -------------------------
    ai_output = AIOutput(
        query=query,
        retrieved_context=retrieved,
        generated_response=response_text,
        recommendations=recs
    )

    # -------------------------
    # Build Readable Output String
    # -------------------------
    divider = "─" * 60
    formatted_output = [f"{divider}\nQUERY:\n{query}\n{divider}"]

    # Retrieved chunks
    formatted_output.append("RETRIEVED CONTEXT:\n")
    for i, chunk in enumerate(retrieved, start=1):
        snippet = chunk.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        formatted_output.append(f"  [{i}] {snippet}\n")

    formatted_output.append(f"{divider}\nAI-GENERATED RESPONSE:\n{response_text.strip()}\n{divider}")

    # Recommendations
    if recs:
        formatted_output.append("STRUCTURED RECOMMENDATIONS:\n")
        for i, rec in enumerate(recs, start=1):
            formatted_output.append(
                f"  {i}. {rec.therapy_type}\n"
                f"     Details: {rec.details}\n"
                f"     Level of Evidence: {rec.level_of_evidence or 'Not specified'}\n"
            )
    else:
        formatted_output.append("No structured recommendations parsed.")

    formatted_output.append(divider)

    return ai_output, "\n".join(formatted_output)


# ==============================
#   STREAMLIT UI
# ==============================
st.set_page_config(page_title="NCCN Clinical Recommendation Assistant", layout="wide", page_icon=":material/oncology:")
st.title(" NCCN RAG Clinical Recommendation Assistant")

with st.form("Patient Data Input"):
    # Patient Details
    st.subheader(" Patient Details")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    perf_status = st.text_input("Performance Status (e.g., ECOG 0)")
    comorbidities = st.text_area("Comorbidities (comma-separated)")
    extra_notes = st.text_area("Other clinical details (optional)", value="")  

# Tumor Details
    st.subheader(" Tumor Details")
    site = st.text_input("Tumor Site", value="oral tongue")
    histology = st.text_input("Histology", value="Carcinoma")
    stage = st.text_input("Stage", value="T1N0M0")
    molecular_features = st.text_input("Molecular Features (optional)", value="")
    imaging_findings = st.text_area("Imaging Findings (optional)", value="")
    
# Prior Therapy
    st.subheader(" Prior Therapy")
    prior_given = st.checkbox("Prior therapy given?")
    therapy_type = st.text_input("Therapy Type") #if prior_given else None
    therapy_details = st.text_input("Therapy Details") #if prior_given else None
    submitted = st.form_submit_button(" Get Recommendations")

if submitted:
    # Load FAISS artifacts
    index, embeddings, text_chunks = load_artifacts()

    # Build patient object
    patient = Patient(
        age=age if age > 0 else None,
        gender=gender if gender else None,
        performance_status=perf_status if perf_status else None,
        comorbidities=[c.strip() for c in comorbidities.split(",")] if comorbidities else None,
        additional_notes=extra_notes if extra_notes else None,  # NEW ✅
    )

    # Build tumor object
    tumor = Tumor(
        site=site,
        histology=histology,
        stage=stage,
        molecular_features=molecular_features if molecular_features else None,
        imaging_findings=imaging_findings if imaging_findings else None,  # NEW ✅
    )

    # Prior therapy object
    prior_therapy = (
        PriorTherapy(therapy_type=therapy_type, details=therapy_details)
        if prior_given else None
    )
    start_time = time.time()
    # Build query and run RAG
    query = build_query_from_structured_input(patient, tumor, prior_therapy)
    retrieved_chunks = retrieve_chunks(query, index, text_chunks)
    response_text = generate_rag_response(query, retrieved_chunks)
    processing_time = time.time() - start_time
    # Get both structured and formatted outputs
    parsed_output, formatted_text = parse_ai_response(query, retrieved_chunks, response_text)

    # Display results in a neat format
    st.subheader(" Formatted AI Output")
    st.text_area("AI Output", formatted_text, height=1000)
    st.caption(f"⏱️ Response generated in **{processing_time:.2f} seconds**")

    # Optional: still show structured recommendations separately if you want
    # st.subheader("💡 Structured Recommendations")
    # if parsed_output.recommendations:
    #     for i, rec in enumerate(parsed_output.recommendations, start=1):
    #         st.markdown(f"**{i}. {rec.therapy_type}**")
    #         st.write(rec.details)
    #         if rec.level_of_evidence:
    #             st.caption(f" Level of Evidence: {rec.level_of_evidence}")
    # else:
    #     st.info("No structured recommendations parsed.")
