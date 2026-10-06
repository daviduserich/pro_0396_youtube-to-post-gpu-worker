# ACP Pro0396 — YouTube-to-Post GPU Worker

**Projekt:** Authentic Content Pilot (`Pro0396`)  
**Komponente:** Dedizierter GPU Inferenz-Worker für die YouTube-to-Post Pipeline  
**Modelle:** PyAnnote 3.1 (Speaker Diarization) + OpenAI Whisper Large V3 (Speech-to-Text mit Wort-Timestamps)  
**Infrastruktur:** Hugging Face Dedicated GPU Endpoint (Nvidia A10G)

---

## 1. Übersicht

Dieser Container dient als isolierter Inferenz-Worker für die YouTube-to-Post Pipeline des Authentic Content Pilot (`Pro0396`).
Er entkoppelt die YouTube-Verarbeitung vollständig von den Vertriebs- und Meeting-Transkriptionen aus `Pro_0361` / `Pro_0338`.

### Kernfunktionen
* **Twin-Standardization:** 16 kHz Mono PCM WAV Vorverarbeitung
* **Global Diarization:** `pyannote/speaker-diarization-3.1` zur Sprecher-Identifikation
* **Transcribe Aligned:** `openai/whisper-large-v3` mit Wort-Level Timestamps und 3-Gang VRAM-Gangschaltung
* **Magnet- & Sticky-Matching:** Mathematische Zuordnung von Wörtern zu Sprechern (`MAGNET_TOLERANCE = 0.2s`, `GAP_FILLING_LIMIT = 0.1s`)

---

## 2. API Endpunkte

* `GET /health`: Health-Check & GPU-Status
* `POST /submit`: Asynchroner Job-Submit (multipart/form-data)
  * `mode = "global_diarization"`: Liefert Sprecher-Timeline
  * `mode = "transcribe_aligned"`: Liefert Wort-Transkript für einen 5-Minuten-Chunk
* `GET /status/{job_id}`: Polling des Job-Status (`queued`, `processing`, `done`, `error`)

---

## 3. Build & Deployment

Das Docker-Image wird via GitHub Actions gebaut und in der GitHub Container Registry (`ghcr.io`) veröffentlicht:
* Image: `ghcr.io/daviduserich/pro_0396_youtube-to-post-gpu-worker:latest`
* Tag: `nextgen`
