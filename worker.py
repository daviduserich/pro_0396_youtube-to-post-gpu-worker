import argparse
import os
import logging
import sys
import json
import time
import psutil
import torch
import torchaudio
import numpy as np
import librosa
import gc

# --- KONFIGURATION ---
DEBUG_MODE = True  # Setze auf False für Produktion
# ---------------------

# --- CRITICAL FIX: Numpy Patch für Pyannote ---
if not hasattr(np, "NAN"):
    np.NAN = np.nan

from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("WORKER_FINAL_MAGNET_FIX")

HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def debug_log(msg):
    """Hilfsfunktion für Debugging"""
    if DEBUG_MODE:
        logger.info(f"🕵️ [DEBUG] {msg}")

def format_timestamp(seconds):
    try:
        if seconds is None: return "00:00:00"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    except:
        return "00:00:00"

def log_resource_usage(stage=""):
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        ram_gb = mem_info.rss / (1024 ** 3)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"📊 [STATS] {stage} | RAM: {ram_gb:.2f}GB")
    except:
        pass

# --- NOTARZT-MODUS (Backup) ---
def emergency_segmentation_rescue(asr_pipe, audio_input, sample_rate=16000):
    logger.info("   ☠️ NOTARZT-MODUS WIRD AKTIVIERT...")
    try:
        if isinstance(audio_input, str):
            waveform, sr = torchaudio.load(audio_input)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            audio_array = waveform.squeeze().numpy()
        else:
            audio_array = audio_input

        chunk_size = 30 * sample_rate
        rescued_chunks = []
        
        for i in range(0, len(audio_array), chunk_size):
            segment_audio = audio_array[i : i + chunk_size]
            offset_sec = i / sample_rate
            try:
                # Wichtig: "array" Key nutzen
                out = asr_pipe(
                    {"array": segment_audio, "sampling_rate": 16000}, 
                    return_timestamps="word", 
                    generate_kwargs={"task": "transcribe", "max_new_tokens": 400, "num_beams": 1}
                )
                seg_words = out.get("chunks", [])
                for word in seg_words:
                    w_start, w_end = word["timestamp"]
                    word["timestamp"] = (w_start + offset_sec, w_end + offset_sec)
                rescued_chunks.extend(seg_words)
            except:
                pass
        return {"chunks": rescued_chunks}
    except Exception as e:
        logger.error(f"   ☠️ Notarzt gescheitert: {e}")
        raise e

# --- KERNSYSTEM: Transkription & Alignment ---
def transcribe_aligned(audio_path, timeline_segment, chunk_index):
    logger.info(f">>> [PHASE 2] High-Quality Transkription (Chunk {chunk_index})...")
    
    # --- DEBUGGING: TIMELINE PRÜFUNG ---
    if DEBUG_MODE:
        logger.info(f"🔍 [DEBUG-START] Analysiere Timeline-Input für Chunk {chunk_index}")
        logger.info(f"   -> Anzahl Segmente in Timeline: {len(timeline_segment)}")
        if len(timeline_segment) > 0:
            first_start = timeline_segment[0]['start']
            logger.info(f"   -> Erstes Segment Startzeit: {first_start:.2f}s")
            
            # Check: Senden wir versehentlich lokale Zeiten?
            if chunk_index > 0 and first_start < 100.0:
                 logger.warning(f"   ⚠️ ALARM: n8n sendet LOKALE Zeiten ({first_start}s)! "
                                f"Wir brauchen GLOBALE Zeiten (ca. > {chunk_index * 270}s).")
    # -----------------------------------

    log_resource_usage(f"Start Transcribe Chunk {chunk_index}")
    
    # OFFSET-KONFIGURATION
    STEP_SIZE = 270.0 
    GLOBAL_OFFSET = chunk_index * STEP_SIZE

    # 1. LIBROSA LOADER (Der 8kHz -> 16kHz Fix)
    try:
        logger.info("   🎧 PROCESSING: Lade Audio & erzwinge 16kHz via Librosa...")
        audio_array, _ = librosa.load(audio_path, sr=16000)
        
        # TACHO-CHECK
        samples_count = len(audio_array)
        duration_check = samples_count / 16000.0
        
        logger.info(f"   📏 DATA-CHECK: Array Größe: {samples_count} Samples")
        logger.info(f"   ⏱️ TIME-CHECK: Dauer: {duration_check:.2f}s (Global Offset: {GLOBAL_OFFSET}s)")
        
        if duration_check < 1.0:
            logger.error("   🚨 ALARM: Audiodatei scheint leer!")
            return {"generated_text": "[... FEHLER: Audio zu kurz ...]"}
            
    except Exception as e:
        logger.error(f"   ❌ Fehler beim Laden mit Librosa: {e}")
        raise e

    # 2. PIPELINE INIT
    try:
        asr_pipe = hf_pipeline(
            "automatic-speech-recognition", 
            model=os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3"), 
            chunk_length_s=30, 
            device=DEVICE, 
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
    except Exception as e:
        logger.error(f"Model Init Error: {e}")
        raise e
    
    result = None
    
    # --- GANGSCHALTUNG (VRAM-Schutz & Stabilität optimiert) ---
    # 1. Gang: Beam 1, Batch 2 (Sehr schnell, extrem stabil, spart 70% VRAM)
    # 2. Gang: Beam 1, Batch 1 (Maximale Isolation, Null OOM-Gefahr)
    # 3. Gang: Beam 2, Batch 1 (Fallback mit minimalem Beam)
    ATTEMPTS_CONFIG = [
        (1, 2, "🚀 1. GANG: EFFIZIENT (Beam 1, Batch 2)"),
        (1, 1, "🛡️ 2. GANG: SAFE (Beam 1, Batch 1)"),
        (2, 1, "🏎️ 3. GANG: FALLBACK BEAM 2 (Beam 2, Batch 1)")
    ]

    for beam_size, batch_sz, desc in ATTEMPTS_CONFIG:
        try:
            logger.info(f"   🔄 Schalte in: {desc}...")
            # WICHTIG: Key ist "array"
            result = asr_pipe(
                {"array": audio_array, "sampling_rate": 16000}, 
                batch_size=batch_sz, 
                generate_kwargs={"task": "transcribe", "num_beams": beam_size, "condition_on_prev_tokens": True, "max_new_tokens": 400}, 
                return_timestamps="word"
            )
            logger.info(f"   ✅ Erfolg mit {desc}!")
            break 
        except Exception as e:
            logger.warning(f"   ⚠️ FEHLER bei {desc}: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue 
    
    if result is None:
        try: 
            result = emergency_segmentation_rescue(asr_pipe, audio_array)
        except Exception as rescue_err: 
            del asr_pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"generated_text": f"*** ⚠️ FATALER FEHLER: Chunk {chunk_index} ({rescue_err}) ***"}

    # Explizite Freigabe des Modells & VRAMs nach Transkription
    del asr_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. MATCHING & STICKY LOGIC
    words = result.get("chunks", [])
    
    final_transcript = []
    current_speaker = None
    current_buffer = []
    block_start_time = 0.0
    last_word_end = 0.0
    
    # --- EINSTELLUNGEN: PRÄZISION (Scharf gestellt!) ---
    # Radius für Treffer (Muss klein sein!)
    MAGNET_TOLERANCE = 0.2   
    
    # Klebstoff für Lücken (Muss minimal sein!)
    GAP_FILLING_LIMIT = 0.1  
    # ---------------------------------------------------
    
    timeline_segment = sorted(timeline_segment, key=lambda x: x["start"])

    def get_next_speaker_start(current_time, timeline):
        for seg in timeline:
            if seg["start"] > current_time: return seg
        return None

    if words and words[0].get("timestamp"):
        w_s, _ = words[0]["timestamp"]
        if w_s is not None: block_start_time = w_s + GLOBAL_OFFSET

    for i, word_obj in enumerate(words):
        word_text = word_obj.get("text", "")
        if word_obj.get("timestamp") is None:
            if word_text: current_buffer.append(word_text)
            continue
            
        w_start, w_end = word_obj["timestamp"]
        if w_start is None or w_end is None: continue

        # Globalisierung der Wort-Zeit
        global_w_start = w_start + GLOBAL_OFFSET
        global_w_end = w_end + GLOBAL_OFFSET
        global_w_mid = (global_w_start + global_w_end) / 2.0
        
        best_speaker = None
        match_found = False
        match_method = "None"
        
        # A. Präzise
        for seg in timeline_segment:
            if seg["start"] <= global_w_mid <= seg["end"]:
                best_speaker = seg["speaker"]
                match_found = True
                match_method = "Hit"
                break
        
        # B. Magnet
        if not match_found:
            for seg in timeline_segment:
                if (seg["start"] - MAGNET_TOLERANCE) <= global_w_mid <= (seg["end"] + MAGNET_TOLERANCE):
                    best_speaker = seg["speaker"]
                    match_found = True
                    match_method = "Magnet"
                    break

        # C. Klebstoff (Stark reduziert für bessere Wechsel-Erkennung)
        if not match_found:
            if current_speaker is not None:
                dist_from_last = global_w_start - (last_word_end + GLOBAL_OFFSET)
                
                # Nur kleben, wenn die Lücke winzig ist
                if dist_from_last < GAP_FILLING_LIMIT:
                    next_seg = get_next_speaker_start(global_w_end, timeline_segment)
                    if next_seg:
                         dist_to_next = next_seg["start"] - global_w_end
                         if dist_to_next < dist_from_last:
                             best_speaker = next_seg["speaker"]
                             match_method = "Sticky-Switch"
                         else:
                             best_speaker = current_speaker
                             match_method = "Sticky-Hold"
                    else:
                        best_speaker = current_speaker
                        match_method = "Sticky-Hold"
                    
                    if best_speaker: match_found = True

        if not best_speaker:
            best_speaker = "Unknown"
            match_method = "Fallback"

        # Logging
        if DEBUG_MODE and (best_speaker != current_speaker or best_speaker == "Unknown"):
             debug_log(f"'{word_text.strip()}' ({global_w_mid:.2f}s) -> {best_speaker} [{match_method}]")

        last_word_end = w_end 

        if best_speaker != current_speaker:
            if current_speaker is not None:
                text_block = "".join(current_buffer).strip()
                if text_block:
                    ts_start = format_timestamp(block_start_time)
                    ts_end = format_timestamp(global_w_start)
                    final_transcript.append(f"**[{current_speaker} | {ts_start} - {ts_end}]:**\n{text_block}")
            
            current_buffer = []
            current_speaker = best_speaker
            block_start_time = global_w_start
            
        current_buffer.append(word_text)
        
    if current_buffer and current_speaker:
        text_block = "".join(current_buffer).strip()
        ts_start = format_timestamp(block_start_time)
        ts_end = format_timestamp(last_word_end + GLOBAL_OFFSET)
        final_transcript.append(f"**[{current_speaker} | {ts_start} - {ts_end}]:**\n{text_block}")
    
    formatted_words = []
    for w in words:
        ts = w.get("timestamp")
        if ts and ts[0] is not None and ts[1] is not None:
            formatted_words.append({
                "word": w.get("text", "").strip(),
                "start": round(ts[0] + GLOBAL_OFFSET, 3),
                "end": round(ts[1] + GLOBAL_OFFSET, 3)
            })

    return {
        "generated_text": "\n\n".join(final_transcript) if final_transcript else "[... Stille ...]",
        "words": formatted_words
    }

def run_global_diarization(audio_path, num_speakers=None):
    pass 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["global_diarization", "transcribe_aligned"])
    parser.add_argument("--timeline_path", help="Pfad zum JSON mit der Timeline")
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--num_speakers", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.mode == "global_diarization":
            result = run_global_diarization(args.input, num_speakers=args.num_speakers)
        elif args.mode == "transcribe_aligned":
            if not args.timeline_path: raise ValueError("Timeline Path benötigt!")
            with open(args.timeline_path, "r") as f: timeline_data = json.load(f)
            result = transcribe_aligned(args.input, timeline_data, args.chunk_index)
        with open(args.output, "w") as f: json.dump(result, f)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
