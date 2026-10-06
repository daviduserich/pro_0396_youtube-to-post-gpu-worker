import torch
import logging
import os
import subprocess
import json
import uuid
import asyncio
import time
import gc
import numpy as np

# --- CRITICAL FIX: Numpy Patch VOR Imports ---
if not hasattr(np, "NAN"):
    np.NAN = np.nan

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from pyannote.audio import Pipeline

# WICHTIG: Wir importieren die Transkriptions-Funktion aus dem Worker!
# Stellen Sie sicher, dass worker.py im selben Ordner liegt.
try:
    from worker import transcribe_aligned
except ImportError:
    # Fallback für lokale Tests ohne worker.py
    def transcribe_aligned(*args, **kwargs):
        return {"error": "worker.py not found"}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MANAGER_DISPATCHER")

app = FastAPI()

# --- GLOBALE JOB-LISTE ---
JOBS = {}

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN fehlt! Pyannote wird nicht funktionieren.")

def sanitize_input_audio(input_path):
    """TWIN-STANDARDIZATION: Konvertierung in WAV 16kHz Mono."""
    try:
        output_path = input_path.rsplit(".", 1)[0] + "_input.wav"
        logger.info(f"🧹 Starte Audio-Sanitizing für: {input_path}")
        command = ["ffmpeg", "-y", "-i", input_path, "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(output_path): raise FileNotFoundError("FFmpeg failed")
        logger.info(f"✨ Audio sauber: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Fehler Audio-Sanitizing: {e}")
        return input_path

def process_job_background(job_id: str, file_path: str, mode: str, num_speakers: int = None, timeline_data: list = None, chunk_index: int = 0):
    """
    Der DISPATCHER: Entscheidet basierend auf 'mode', was zu tun ist.
    """
    logger.info(f"⚙️ Starte Job {job_id} | Modus: {mode}")
    JOBS[job_id]["status"] = "processing"
    clean_audio_path = None
    
    try:
        # 1. Immer erst Audio waschen (Twin-Standardization)
        clean_audio_path = sanitize_input_audio(file_path)
        
        # --- WEICHE: Was soll getan werden? ---
        
        if mode == "global_diarization":
            # PHASE 1: WER SPRICHT WANN?
            logger.info("   👉 Starte Pyannote Diarization...")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
            pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            try: pipeline.segmentation_batch_size = 32
            except: pass

            if num_speakers and num_speakers > 0:
                try:
                    diarization = pipeline(clean_audio_path, num_speakers=num_speakers)
                except Exception as dia_err:
                    logger.warning(f"⚠️ Diarization mit num_speakers={num_speakers} fehlgeschlagen ({dia_err}), versuche Fallback ohne feste Sprecheranzahl...")
                    diarization = pipeline(clean_audio_path)
            else:
                diarization = pipeline(clean_audio_path) # Default Parameter

            timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                timeline.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                
            JOBS[job_id]["result"] = {"timeline": timeline}
            
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif mode == "transcribe_aligned":
            # PHASE 2: WAS WIRD GESAGT?
            logger.info(f"   👉 Starte Whisper Transcription (Chunk {chunk_index})...")
            
            if not timeline_data:
                raise ValueError("Timeline fehlt für Transkription!")
            
            # Wir rufen die Funktion aus worker.py auf
            result = transcribe_aligned(clean_audio_path, timeline_data, chunk_index)
            
            # Das Ergebnis ist direkt das JSON mit "generated_text"
            JOBS[job_id]["result"] = result

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        else:
            raise ValueError(f"Unbekannter Modus: {mode}")

        # Abschluss
        JOBS[job_id]["status"] = "done"
        logger.info(f"✅ Job {job_id} erfolgreich beendet.")

    except Exception as e:
        err_str = str(e)
        logger.error(f"❌ Job {job_id} fehlgeschlagen: {err_str}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = err_str

        # Root-Cause-2 Schutz: CUDA Context Corruption
        # Wenn ein CUDA Assert/OOM die Treiberumgebung korrumpiert hat,
        # darf der Container nicht weiter im korrumpierten Zustand verharren.
        # Wir warten kurz (damit n8n das 'error' Status-Polling noch lesen kann)
        # und beenden dann den Python-Prozess hart via os._exit(1).
        # Hugging Face Spaces startet den Container sofort sauber neu.
        if "CUDA" in err_str or "cuda" in err_str or "device-side assert" in err_str:
            logger.critical("🔥 KRITISCHER CUDA-FEHLER ERKANNT! Initiiere Container-Neustart via os._exit(1) in 3s...")
            time.sleep(3)
            os._exit(1)

    finally:
        # Aufräumen von Zwischendateien
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass
        if clean_audio_path and os.path.exists(clean_audio_path) and clean_audio_path != file_path:
            try: os.remove(clean_audio_path)
            except: pass

@app.get("/health")
def health_check():
    return {"status": "running", "gpu": torch.cuda.is_available()}

@app.post("/submit")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    mode: str = Form("global_diarization"),
    num_speakers: int = Form(None),
    timeline_json: str = Form(None), # NEU: Timeline Empfang
    chunk_index: int = Form(0)       # NEU: Chunk Index
):
    job_id = str(uuid.uuid4())
    logger.info(f"📥 Job empfangen: {job_id} | Mode: {mode}")
    
    # Timeline parsen (kommt als JSON-String von n8n)
    timeline_data = None
    if timeline_json:
        try:
            timeline_data = json.loads(timeline_json)
        except:
            logger.warning("Konnte Timeline-JSON nicht parsen.")

    # Datei speichern
    temp_filename = f"/tmp/{job_id}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())
    
    JOBS[job_id] = {"status": "queued", "mode": mode}
    
    # Background Task starten (mit allen neuen Parametern)
    background_tasks.add_task(
        process_job_background, 
        job_id, 
        temp_filename, 
        mode, 
        num_speakers, 
        timeline_data, 
        chunk_index
    )
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID nicht gefunden")
    return JOBS[job_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
