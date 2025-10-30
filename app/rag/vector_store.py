"""Utilities for interacting with the Chroma vector store and BM25 fallback."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
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

_DEFAULT_CORPUS: List[Dict[str, str]] = [
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


def _compute_dense_score(distance: float) -> float:
    """
    Convert distance to similarity score.
    Handles different distance metrics (L2, cosine, etc.)
    """
    if distance <= 0:
        return 1.0
    return 1.0 / (1.0 + distance)


def _bm25_search(
    query: str,
    *,
    corpus: Sequence[Dict[str, str]],
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Execute BM25 search and return a mapping of document ID to normalized score.
    
    Returns:
        Dict mapping document ID to BM25 score (0-1 range)
    """
    if not corpus:
        return {}
    
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores_raw = bm25.get_scores(query.lower().split())
    scores_list = (
        scores_raw.tolist()
        if hasattr(scores_raw, "tolist")
        else list(scores_raw)
    )

    if not scores_list:
        return {}

    max_score = max(scores_list)

    result = {}
    for doc, score in zip(corpus, scores_list):
        normalized_score = float(score / max_score) if max_score > 0 else 0.0
        result[doc["id"]] = normalized_score
    
    return result


def search(
    query_text: str,
    *,
    top_k: int = 4,
    hyde_text: Optional[str] = None,
    extra_corpus: Optional[Sequence[Dict[str, str]]] = None,
    embedder: Optional[ChatProvider] = None,
) -> List[Dict[str, object]]:
    """
    Hybrid search combining Dense (vector) and BM25 (keyword) on unified corpus.
    
    Strategy:
    1. Retrieve top_k*3 candidates from Chroma (vector search)
    2. Run BM25 on the same candidates
    3. Merge scores with weighted average
    4. Return top_k results
    """
    embedder = embedder or get_llm_provider()
    col = get_collection()
    
    candidate_multiplier = 3  
    n_candidates = max(top_k * candidate_multiplier, 20)  # 至少获取20个
    
    candidates: List[Dict[str, object]] = []
    
    try:
        query_embedding = embedder.embed([hyde_text or query_text])[0]
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        
        for doc_id, doc, meta, distance in zip(ids, docs, metas, distances):
            dense_distance = float(distance) if distance is not None else 1.0
            dense_score = _compute_dense_score(dense_distance)
            
            candidates.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta or {},
                "dense_score": dense_score,
                "dense_distance": dense_distance,
                "source": meta.get("source") if isinstance(meta, dict) else doc_id,
            })
        
        logger.info("Dense search retrieved %d candidates from Chroma", len(candidates))
        
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector search failed: %s", exc)
        candidates = [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "dense_score": 0.0,
                "dense_distance": None,
                "source": doc["id"],
            }
            for doc in _DEFAULT_CORPUS
        ]
        logger.info("Using default corpus fallback (%d documents)", len(candidates))
    
    if not candidates:
        logger.warning("No candidates found, returning default corpus")
        return [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "hybrid_score": 0.0,
                "bm25_raw_score": 0.0,
                "dense_distance": None,
                "score": 0.0,
                "source": doc["id"],
            }
            for doc in _DEFAULT_CORPUS[:top_k]
        ]
    
    bm25_corpus = [
        {"id": doc["id"], "text": doc["text"]}
        for doc in candidates
    ]
    
    bm25_scores = _bm25_search(query_text, corpus=bm25_corpus, top_k=len(candidates))
    logger.info("BM25 search computed scores for %d documents", len(bm25_scores))
    
    results = []
    for candidate in candidates:
        doc_id = candidate["id"]
        dense_score = candidate["dense_score"]
        bm25_score = bm25_scores.get(doc_id, 0.0)
        
        hybrid_score = (dense_score * 0.6) + (bm25_score * 0.4)
        
        results.append({
            "id": doc_id,
            "text": candidate["text"],
            "metadata": candidate["metadata"],
            "source": candidate["source"],
            "dense_score": round(dense_score, 4),
            "bm25_score": round(bm25_score, 4),
            "hybrid_score": round(hybrid_score, 4),
            "score": round(hybrid_score, 4),  
            "dense_distance": candidate.get("dense_distance"),
            "bm25_raw_score": None, 
        })
    
    ranked = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)
    
    logger.info(
        "Hybrid search returned %d results (top_%d requested)",
        len(ranked[:top_k]),
        top_k,
    )
    
    return ranked[:top_k]


def ensure_seed_documents(embedder: Optional[ChatProvider] = None) -> None:
    """Ensure the default demo corpus exists in the vector store."""
    col = get_collection()
    seed_ids = [doc["id"] for doc in _DEFAULT_CORPUS]
    existing_ids: set[str] = set()
    
    try:
        existing = col.get(ids=seed_ids)
        id_batches = existing.get("ids") or []
        for batch in id_batches:
            if isinstance(batch, list):
                existing_ids.update(batch)
            elif isinstance(batch, str):
                existing_ids.add(batch)
    except Exception:  # noqa: BLE001
        existing_ids = set()
    
    new_docs = [doc for doc in _DEFAULT_CORPUS if doc["id"] not in existing_ids]
    if not new_docs:
        logger.info("All seed documents already exist in vector store")
        return
    
    embedder = embedder or get_llm_provider()
    metadatas = [
        {
            "source": doc["id"],
            "type": "seed_job",
            "created_at": datetime.utcnow().isoformat(),
        }
        for doc in new_docs
    ]
    
    try:
        texts = [doc["text"] for doc in new_docs]
        embeddings = embedder.embed(texts)
        col.add(
            ids=[doc["id"] for doc in new_docs],
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Seeded %d job documents into the vector store.", len(new_docs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seeding default corpus failed: %s", exc)
