# PRD — Intracranial Aneurysm Detector (Research Web App)

> **Status:** Draft v1
> **Owner:** You
> **Purpose:** Turn the existing research pipeline (models + segmentation + post-processing) into a mobile-friendly web app for **upload → instant analysis → report**.
> **Regulatory note:** This is a **research/education** tool, **not a medical device** or diagnostic aid.

---

## 0) Executive Summary

Build a responsive web app (desktop + mobile) that lets users upload **PNG/JPEG/DICOM (.dcm or .zip of a series)** and receive an **instant research report** with: candidate aneurysm detections (boxes + scores), key slice thumbnails, and optional downloadable artifacts. A **fast path** returns results in seconds using the detector + anatomical post-processing; a **full path** optionally triggers heavy segmentation as a background job.

---

## 1) Goals & Success Metrics

**Goals**

* Provide a 1-click, mobile-friendly **upload → analyze → view report** flow.
* Support **DICOM** (single file or zipped series) + **JPEG/PNG**.
* Return **instant JSON results** and **thumbnail overlays**; optional **PDF** result.
* Clear **privacy** and **disclaimer** UX.

**Success Metrics**

* TTFI (time to first inference result) ≤ **10s** for medium studies on GPU (fast mode).
* ≥ **95%** successful parses for valid PNG/JPEG/DICOM inputs.
* End-to-end errors < **2%** in MVP testing dataset.
* Mobile test pass on iOS Safari & Android Chrome (upload + view).

---

## 2) Scope / Non-Goals

**In Scope**

* File upload UI, progress, results view.
* DICOM handling (stack, rescale slope/intercept, basic de-ID).
* REST API (FastAPI) with sync analyze endpoint; optional async job for heavy segmentation.
* Simple overlays (2–5 key slices) + JSON report; optional PDF export.
* Dockerized dev; GPU runtime support.

**Out of Scope**

* Clinical validation, CE/FDA compliance.
* PACS/HL7 integration.
* Full DICOM web viewer (OHIF) in MVP (can be Phase 2).
* User accounts & billing (Phase ≥2).

---

## 3) Users & Personas

* **Researcher / Engineer**: uploads anonymized scans to compare model versions.
* **Student / Demonstrator**: shows research pipeline results in class or demo.
* **Clinically curious user**: explores how the model behaves (must see “research only” disclaimer).

---

## 4) User Stories

1. **As a user**, I can upload **PNG/JPEG/DICOM (.dcm or .zip)** and get an **instant report** with candidate boxes and scores.
2. **As a user**, I can **toggle** “Full (segmentation)” vs “Fast (detector-only)” modes (default Fast).
3. **As a user**, I can **download** artifacts (JSON, overlays; PDF optional).
4. **As a user**, I see **clear errors** if input is invalid and suggested fixes.
5. **As a user**, I see a **privacy/disclaimer** notice before upload.

---

## 5) UX Flows

### 5.1 Upload Flow (Fast)

1. Landing: “Research only” banner; file drop; tips (ZIP DICOM for best results).
2. Select file → progress bar → call `POST /analyze?use_seg=0`
3. Show results: candidate list, scores, 2–5 slice thumbnails, download JSON.

### 5.2 Upload Flow (Full)

1. Toggle “Full (segmentation)” → `POST /jobs` returns `job_id`
2. UI polls `GET /jobs/{id}` until **complete**; show refined results.

### 5.3 Errors

* Unsupported format → explain accepted formats.
* DICOM parse fail → suggest zipping series, check transfer syntax.
* Server error → show error code and feedback link.

---

## 6) Functional Requirements

**Front-end (Next.js/React)**

* Responsive layout (mobile first).
* File input with accept: `.png,.jpg,.jpeg,.dcm,.zip`
* Toggle for “Full” mode (disabled by default).
* Results panel: summary, candidate table, thumbnails.
* Download buttons: JSON; (Phase 2: PDF).
* Status & error toasts.

**Back-end (FastAPI)**

* **Sync** endpoint `/analyze` for detector + post-processing.
* (Phase 2) **Async** job endpoints for heavy segmentation.
* DICOM parsing and **basic de-identification**.
* Overlays: render bounding boxes on slices; save to `/media/<case>/sliceNNN.png`.
* Return JSON with candidates, links, notes.
* Health & version endpoints.

**DICOM Handling**

* Accept single `.dcm` or `.zip` of multiple DICOMs.
* Assemble 3D volume sorted by `ImagePositionPatient` or `InstanceNumber`.
* Apply `RescaleSlope/Intercept` to HU; preserve `PixelSpacing`/`SliceThickness`.
* **Basic de-ID**: remove fields like `PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `InstitutionName`.

---

## 7) Non-Functional Requirements

* **Performance**: Fast path ≤ 10s typical on mid GPU; CPU fallback allowed with warning.
* **Reliability**: Robust parsing with clear messages.
* **Security**: HTTPS in prod; CORS restricted; max upload size configurable; virus scan optional.
* **Privacy**: de-ID on ingest; default ephemeral storage (≤72h) with immediate delete option.
* **Observability**: structured logs, basic metrics (requests, durations, error counts).

---

## 8) Architecture & Tech

* **Frontend**: Next.js + Tailwind (or plain CSS), fetch API.
* **Backend**: FastAPI (Uvicorn), PyTorch, pydicom, Pillow, numpy.
* **Container**: Docker, optional NVIDIA runtime for GPU.
* **Storage**: local `./storage` for MVP; S3 in Phase 2.
* **Background jobs** (Phase 2): RQ/Celery + Redis for segmentation.

---

## 9) API Spec (MVP)

**POST /analyze**

* **Form**: `file` = PNG/JPG/JPEG/DICOM/ZIP
* **Query**: `use_seg=0|1` (default 0. If 1 and workers disabled, return 501).
  **Response 200**

```json
{
  "n_candidates": 2,
  "candidates": [
    {"bbox":[z1,y1,x1,z2,y2,x2], "score":0.82},
    {"bbox":[...], "score":0.67}
  ],
  "thumbnails": [
    {"slice":142, "url":"/media/abc/slice142.png"},
    {"slice":159, "url":"/media/abc/slice159.png"}
  ],
  "notes":"Research use only — not for clinical diagnosis."
}
```

**Errors**

* 400: unsupported or corrupt file
* 413: payload too large
* 500: internal error (include `error_id`)

**GET /health** → `{"status":"ok"}`
**GET /versions** → model name, ckpt hash, pipeline version

**(Phase 2) POST /jobs** (use\_seg=1) → `{ "job_id": "…" }`
\*\*(Phase 2) GET /jobs/{id}`** → `{"status":"queued|running|done|error", "result":{…}}\`

---

## 10) Data & Storage

* **Input**: Keep original file for the duration of job; delete on completion or after TTL.
* **Artifacts**: store generated thumbnails/JSON under `/storage/<case_id>/…` (Phase 2: move to S3).
* **Retention**: default TTL 24–72h; user can delete immediately.

---

## 11) Security & Privacy

* **De-identify** basic PHI tags on DICOM ingest.
* Prepend **disclaimer** and consent checkbox in UI.
* **Rate limits** (Phase 2): IP-based to avoid abuse.
* **CORS**: restrict to your frontend origin(s).
* **HTTPS** in prod; secrets via env vars.

---

## 12) Deployment

* **Dev**: docker-compose (api + web).
* **GPU**: set `deploy.resources.reservations.devices.capabilities=[gpu]`.
* **Env Vars**:

  * `MODEL_CKPT` — path to weights
  * `STORAGE_DIR` — output dir
  * `NEXT_PUBLIC_API` — frontend → backend URL
  * `MAX_UPLOAD_MB` — e.g., 1024
  * `ALLOWED_ORIGINS` — CORS origins
  * `ENABLE_SEG` — `0|1`

---

## 13) Roadmap & Milestones

### Milestone 1 — **MVP: Fast Path** (1–2 sprints)

* ✅ `/analyze` (sync) with detector + anatomical filter
* ✅ DICOM assemble + de-ID + PNG/JPG support
* ✅ Overlays & JSON return
* ✅ Responsive upload UI + results list
* ✅ Dockerized dev
  **Definition of Done**
* Upload (png/jpg/dcm/zip) works on desktop + mobile
* Result returns within time budget (GPU)
* Logs show 200/4xx/5xx; errors are human-readable
* README quickstart: works from clean clone

### Milestone 2 — **Full Mode & Reports** (2–3 sprints)

* ☐ Background jobs for segmentation (RQ/Celery)
* ☐ Job polling UI + progress
* ☐ PDF export (simple)
* ☐ S3 storage + signed URLs
* ☐ Basic metrics dashboard
  **Definition of Done**
* Full mode runs; jobs robust to restarts
* PDF download; artifacts signed URLs
* Grafana/Prom logs minimal dashboard

### Milestone 3 — **Viewer & Hardening** (3+)

* ☐ DICOM viewer (CornerstoneJS/OHIF)
* ☐ Auth (simple)
* ☐ Feature flags for model switching (CPM-Net vs 3D-CNN-TR)
* ☐ Load + compare versions
* ☐ Advanced privacy features (consent, purge, audit)

---

## 14) Acceptance Criteria (MVP)

* **AC-1**: Upload any supported format (PNG/JPEG/DICOM/ZIP) → `/analyze` returns `200` with `n_candidates ≥ 0`, `candidates[]`, optional `thumbnails[]`, and disclaimer text.
* **AC-2**: Invalid format returns 400 with helpful message.
* **AC-3**: 10 test studies complete under **10s** (GPU) in fast mode.
* **AC-4**: Mobile Safari & Chrome test: upload works; results render without layout break.
* **AC-5**: DICOM de-ID removes key PHI tags; no PHI persisted in JSON or logs.

---

## 15) Detailed Checklists (for Windsurf / coding sessions)

### 15.1 Backend (FastAPI)

* [ ] Create `api/main.py` with endpoints: `/health`, `/versions`, `/analyze`
* [ ] Add `requirements.txt` (fastapi, uvicorn, pydicom, pillow, numpy, torch)
* [ ] Implement DICOM ingestion:

  * [ ] ZIP support (unzip in memory)
  * [ ] Sort slices; apply slope/intercept; uint16 → HU float
  * [ ] De-identify basic PHI tags
* [ ] Normalization to model input; shape `(B,1,Z,Y,X)`
* [ ] Wire **your** `predict()` and `AnatomicalFilter.run()`
* [ ] Generate overlays: choose 2–5 key slices; draw boxes
* [ ] Return JSON response (see spec)
* [ ] Add `MODEL_CKPT`, `STORAGE_DIR` env usage
* [ ] Logging (request id, duration, file type, success/fail)

**Windsurf prompt idea:**
“Create a FastAPI `/analyze` endpoint that accepts file upload (png/jpg/jpeg/dcm/zip), normalizes to a 3D volume, calls `detector.predict(vol)` (already implemented in my repo), then runs `AnatomicalFilter.run(...)`. Save 3 PNG overlays in `/storage/<case_id>` and return JSON with candidate boxes, scores, and thumbnail URLs.”

### 15.2 Frontend (Next.js)

* [ ] File upload component (accept attr) + size checks
* [ ] Call `/analyze` with FormData; show spinner and progress
* [ ] Results list: candidate index, score, bbox
* [ ] Thumbnails grid, lazy load, tap to enlarge
* [ ] Download buttons for JSON (blob)
* [ ] Error handling (toasts)
* [ ] Mobile CSS test (iPhone/Android)
* [ ] Env var `NEXT_PUBLIC_API`

**Windsurf prompt idea:**
“Build a Next.js page with a drag-and-drop file area accepting .png .jpg .jpeg .dcm .zip; on submit call `${process.env.NEXT_PUBLIC_API}/analyze`; show a loading state and render candidates + thumbnails on success; handle 4xx/5xx with user-friendly messages.”

### 15.3 DevOps

* [ ] `api/Dockerfile` (CUDA runtime if GPU)
* [ ] `web/Dockerfile`
* [ ] `docker-compose.yml` (api:8000, web:3000; volumes for checkpoints/storage)
* [ ] `.env.example` with required vars
* [ ] README quickstart
* [ ] GPU test run (`nvidia-smi` present in container)

---

## 16) Risks & Mitigations

* **Slow segmentation** → make it optional; move to worker in Phase 2.
* **DICOM variability** → robust parsing; helpful error messages; FAQ in README.
* **GPU unavailability** → CPU fallback + warning; batch resize or smaller patches.
* **Privacy concerns** → strict de-ID; retention TTL; explicit disclaimers.

---

## 17) Future Enhancements

* OHIF/CornerstoneJS viewer with scrolling & windowing.
* Multi-model ensembling toggle and comparison charts.
* User auth & private study history.
* Advanced FP analysis overlay (e.g., reason categories).
* Internationalization (UI text).

---

### Appendix A — Sample De-ID Tags to Remove (non-exhaustive)

* `PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `PatientAddress`, `InstitutionName`, `ReferringPhysicianName`, `AccessionNumber`, `StudyID`.
  *(Keep only what’s needed for technical reconstruction; document what you drop.)*

---

## Final Notes for Execution in Windsurf

* Start with **Milestone 1** tasks (backend `/analyze`, frontend upload page, Docker).
* Use the **checklists** as task prompts; commit after each check passes.
* Keep the **MVP** small: detector + post-processing only; segmentation later.
* After MVP works locally, consider **S3 + signed URLs** and **PDF** in Phase 2.

If you want, I can turn this PRD into a **/docs/PRD.md** and add **/api**, **/web**, and **docker** scaffolds so you can start coding immediately.
