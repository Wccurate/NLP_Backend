"""Utilities for interacting with the Chroma vector store and BM25 fallback."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from ..config import get_settings
from ..llm_provider import ChatProvider, get_llm_provider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "jobs_demo"

# _DEFAULT_CORPUS: List[Dict[str, str]] = [
#     {
#         "id": "jobs_demo#1",
#         "text": "Software Engineer working on backend APIs with FastAPI and cloud services.",
#     },
#     {
#         "id": "jobs_demo#2",
#         "text": "Machine Learning Engineer focusing on NLP, transformers, and production pipelines.",
#     },
#     {
#         "id": "jobs_demo#3",
#         "text": "Data Analyst role requiring SQL, dashboards, and storytelling with data.",
#     },
#     {
#         "id": "jobs_demo#4",
#         "text": "DevOps Engineer supporting CI/CD, Kubernetes, and infrastructure automation.",
#     },
# ]
_DEFAULT_CORPUS: List[Dict[str, str]] = [
    # --- 科技 (Software & Data) ---
    {
        "id": "jobs_demo#1",
        "text": "Software Engineer working on backend APIs with FastAPI and cloud services.",
    },
    {
        "id": "jobs_demo#2",
        "text": "Machine Learning Engineer focusing on NLP, transformers, and production pipelines.",
    },
    {
        "id": "jobs_demo#3",
        "text": "Data Analyst role requiring SQL, dashboards, and storytelling with data.",
    },
    {
        "id": "jobs_demo#4",
        "text": "DevOps Engineer supporting CI/CD, Kubernetes, and infrastructure automation.",
    },
    {
        "id": "jobs_demo#5",
        "text": "Frontend Developer specializing in React, TypeScript, and modern CSS frameworks.",
    },
    {
        "id": "jobs_demo#6",
        "text": "Data Scientist, PhD required, focusing on causal inference and bayesian modeling.",
    },
    {
        "id": "jobs_demo#7",
        "text": "Mobile App Developer (iOS) proficient in Swift, SwiftUI, and CoreData.",
    },
    {
        "id": "jobs_demo#8",
        "text": "Android Developer with experience in Kotlin, Jetpack Compose, and RxJava.",
    },
    {
        "id": "jobs_demo#9",
        "text": "Cloud Architect (AWS) responsible for designing scalable, serverless microservices.",
    },
    {
        "id": "jobs_demo#10",
        "text": "Cybersecurity Analyst monitoring for threats, SIEM, and vulnerability assessment.",
    },
    {
        "id": "jobs_demo#11",
        "text": "QA Automation Engineer writing test scripts using Selenium, Cypress, and Pytest.",
    },
    {
        "id": "jobs_demo#12",
        "text": "Database Administrator (DBA) managing PostgreSQL clusters and query optimization.",
    },
    {
        "id": "jobs_demo#13",
        "text": "Site Reliability Engineer (SRE) focused on system observability, SLOs, and incident response.",
    },
    {
        "id": "jobs_demo#14",
        "text": "Technical Product Manager defining roadmap for a B2B SaaS platform.",
    },
    {
        "id": "jobs_demo#15",
        "text": "AI Researcher publishing papers on multimodal learning and generative models.",
    },
    {
        "id": "jobs_demo#16",
        "text": "Game Developer using Unity, C#, and 3D graphics programming.",
    },
    {
        "id": "jobs_demo#17",
        "text": "Embedded Systems Engineer programming firmware in C/C++ for IoT devices.",
    },

    # --- 创意与设计 (Creative & Design) ---
    {
        "id": "jobs_demo#18",
        "text": "UX/UI Designer creating wireframes, prototypes in Figma, and conducting user testing.",
    },
    {
        "id": "jobs_demo#19",
        "text": "Graphic Designer for branding, marketing materials, using Adobe Creative Suite.",
    },
    {
        "id": "jobs_demo#20",
        "text": "Content Writer specializing in SEO, long-form blog posts, and technical whitepapers.",
    },
    {
        "id": "jobs_demo#21",
        "text": "Video Editor proficient in Adobe Premiere Pro, After Effects, and color grading.",
    },
    {
        "id": "jobs_demo#22",
        "text": "Animation Artist (3D) using Blender and Maya for character modeling and rigging.",
    },

    # --- 商业与金融 (Business & Finance) ---
    {
        "id": "jobs_demo#23",
        "text": "Digital Marketing Manager overseeing PPC, SEO, and email marketing campaigns.",
    },
    {
        "id": "jobs_demo#24",
        "text": "Sales Development Representative (SDR) doing cold outreach and qualifying leads.",
    },
    {
        "id": "jobs_demo#25",
        "text": "Account Executive (AE) managing the full sales cycle and closing enterprise deals.",
    },
    {
        "id": "jobs_demo#26",
        "text": "Financial Analyst (FP&A) responsible for budgeting, forecasting, and variance analysis.",
    },
    {
        "id": "jobs_demo#27",
        "text": "Accountant (CPA) handling accounts payable, receivable, and financial reporting.",
    },
    {
        "id": "jobs_demo#28",
        "text": "Investment Banking Analyst building financial models and executing M&A deals.",
    },
    {
        "id": "jobs_demo#29",
        "text": "Management Consultant solving strategic problems for Fortune 500 clients.",
    },
    {
        "id": "jobs_demo#30",
        "text": "Business Analyst bridging the gap between stakeholders and the development team.",
    },

    # --- 运营与支持 (Operations & Support) ---
    {
        "id": "jobs_demo#31",
        "text": "Human Resources (HR) Generalist managing payroll, benefits, and employee relations.",
    },
    {
        "id": "jobs_demo#32",
        "text": "Technical Recruiter sourcing candidates for highly specialized engineering roles.",
    },
    {
        "id": "jobs_demo#33",
        "text": "Project Manager (PMP) coordinating timelines, resources, and deliverables.",
    },
    {
        "id": "jobs_demo#34",
        "text": "Agile Coach / Scrum Master facilitating sprint ceremonies and team processes.",
    },
    {
        "id": "jobs_demo#35",
        "text": "Customer Support Specialist handling inbound tickets via Zendesk and live chat.",
    },
    {
        "id": "jobs_demo#36",
        "text": "Supply Chain Manager optimizing logistics, inventory, and vendor negotiations.",
    },
    {
        "id": "jobs_demo#37",
        "text": "Operations Manager overseeing daily business functions and process improvement.",
    },
    {
        "id": "jobs_demo#38",
        "text": "Executive Assistant managing calendars, travel, and communications for C-level.",
    },

    # --- 医疗与科学 (Healthcare & Science) ---
    {
        "id": "jobs_demo#39",
        "text": "Registered Nurse (RN) working in the Intensive Care Unit (ICU).",
    },
    {
        "id": "jobs_demo#40",
        "text": "Medical Researcher (Biotech) conducting experiments on CAR-T cell therapies.",
    },
    {
        "id": "jobs_demo#41",
        "text": "Bioinformatics Scientist analyzing genomic data (NGS) using Python and R.",
    },
    {
        "id": "jobs_demo#42",
        "text": "Clinical Research Coordinator (CRC) managing patient recruitment for trials.",
    },
    {
        "id": "jobs_demo#43",
        "text": "Pharmacist dispensing medication and advising patients on drug interactions.",
    },

    # --- 工程 (非软件) 与其他 (Engineering & Other) ---
    {
        "id": "jobs_demo#44",
        "text": "Mechanical Engineer designing components in SolidWorks and performing FEA.",
    },
    {
        "id": "jobs_demo#45",
        "text": "Electrical Engineer designing PCB layouts for consumer electronics.",
    },
    {
        "id": "jobs_demo#46",
        "text": "Civil Engineer managing large-scale infrastructure projects and site plans.",
    },
    {
        "id": "jobs_demo#47",
        "text": "Aerospace Engineer working on satellite propulsion systems.",
    },
    {
        "id": "jobs_demo#48",
        "text": "Paralegal assisting attorneys with case research and document preparation.",
    },
    {
        "id": "jobs_demo#49",
        "text": "High School Teacher (Mathematics) teaching Algebra and Calculus.",
    },
    {
        "id": "jobs_demo#50",
        "text": "Architect (AIA) designing commercial buildings using AutoCAD and Revit.",
    },
]


@lru_cache()
def _get_client() -> chromadb.api.client.ClientAPI:
    settings = get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client_settings = Settings(anonymized_telemetry=False, allow_reset=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir), settings=client_settings)


@lru_cache()
def get_collection() -> Collection:
    """Return the persistent Chroma collection."""

    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _ensure_ids(count: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(count)]


def add_texts(
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, str]]] = None,
    *,
    ids: Optional[Sequence[str]] = None,
    embedder: Optional[ChatProvider] = None,
) -> List[str]:
    """Add documents to Chroma and return their ids."""

    if not texts:
        return []
    embedder = embedder or get_llm_provider()
    col = get_collection()
    doc_ids = list(ids) if ids else _ensure_ids(len(texts))
    embeddings = embedder.embed(texts)
    col.add(
        ids=doc_ids,
        documents=list(texts),
        metadatas=list(metadatas) if metadatas else None,
        embeddings=embeddings,
    )
    return doc_ids


def delete(ids: Iterable[str]) -> None:
    """Delete documents by id."""

    col = get_collection()
    try:
        col.delete(ids=list(ids))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed deleting ids %s: %s", ids, exc)


def _bm25_search(
    query: str,
    *,
    docs: Optional[Sequence[Dict[str, str]]] = None,
    top_k: int = 4,
) -> List[Dict[str, object]]:
    corpus = list(docs) if docs else _DEFAULT_CORPUS
    if not corpus:
        return []
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    max_score = max(scores) if len(scores) else 0.0
    scored = sorted(
        zip(corpus, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    return [
        {
            "id": entry["id"],
            "text": entry["text"],
            "bm25_raw_score": float(score),
            "bm25_score": float(score / max_score) if max_score > 0 else 0.0,
            "metadata": entry.get("metadata", {}),
            "source": entry["id"],
        }
        for entry, score in scored
    ]


def search(
    query_text: str,
    *,
    top_k: int = 4,
    hyde_text: Optional[str] = None,
    extra_corpus: Optional[Sequence[Dict[str, str]]] = None,
    embedder: Optional[ChatProvider] = None,
) -> List[Dict[str, object]]:
    """Hybrid search combining Chroma and BM25 results."""

    embedder = embedder or get_llm_provider()
    col = get_collection()
    dense_results: List[Dict[str, object]] = []
    try:
        query_embedding = embedder.embed([hyde_text or query_text])[0]
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for doc_id, doc, meta, distance in zip(ids, docs, metas, distances):
            dense_distance = float(distance) if distance is not None else None
            dense_score = None
            if dense_distance is not None:
                dense_score = max(0.0, 1.0 - dense_distance)
            dense_results.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {},
                    "dense_score": float(dense_score) if dense_score is not None else 0.0,
                    "dense_distance": dense_distance,
                    "bm25_score": 0.0,
                    "bm25_raw_score": None,
                    "source": meta.get("source") if isinstance(meta, dict) else doc_id,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector search failed: %s", exc)

    bm25_docs: List[Dict[str, str]] = list(extra_corpus) if extra_corpus else []
    bm25_results = _bm25_search(query_text, docs=bm25_docs or None, top_k=top_k)

    merged: Dict[str, Dict[str, object]] = {}

    def _merge_item(entry: Dict[str, object]) -> None:
        key = entry["id"]
        existing = merged.get(key)
        if existing is None:
            existing = {
                "id": entry["id"],
                "text": entry.get("text", ""),
                "metadata": entry.get("metadata", {}),
                "source": entry.get("source") or entry["id"],
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "dense_distance": None,
                "bm25_raw_score": None,
            }
            merged[key] = existing
        if entry.get("text"):
            existing["text"] = entry["text"]
        if entry.get("metadata"):
            existing["metadata"] = entry["metadata"]
        if entry.get("source"):
            existing["source"] = entry["source"]
        if entry.get("dense_score") is not None:
            existing["dense_score"] = float(entry.get("dense_score") or 0.0)
        if entry.get("dense_distance") is not None:
            existing["dense_distance"] = float(entry["dense_distance"])
        if entry.get("bm25_score") is not None:
            existing["bm25_score"] = float(entry.get("bm25_score") or 0.0)
        if entry.get("bm25_raw_score") is not None:
            existing["bm25_raw_score"] = float(entry["bm25_raw_score"])

    for item in dense_results:
        _merge_item(item)
    for item in bm25_results:
        _merge_item(item)

    for entry in merged.values():
        dense_score = float(entry.get("dense_score") or 0.0)
        bm25_score = float(entry.get("bm25_score") or 0.0)
        hybrid_score = (dense_score * 0.6) + (bm25_score * 0.4)
        entry["hybrid_score"] = hybrid_score
        entry["score"] = hybrid_score

    ranked = sorted(
        merged.values(),
        key=lambda item: item.get("hybrid_score") or 0,
        reverse=True,
    )
    if not ranked:
        fallback = [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "hybrid_score": 0.0,
                "bm25_raw_score": None,
                "dense_distance": None,
                "score": 0.0,
                "source": doc["id"],
            }
            for doc in _DEFAULT_CORPUS[:top_k]
        ]
        logger.info("Hybrid search returned 0 candidates, using default corpus fallback.")
        return fallback

    logger.info("Hybrid search returned %d candidates", len(ranked))
    return ranked[:top_k]
