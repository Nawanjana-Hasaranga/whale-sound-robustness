import os
import shutil
import random
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = "raw_dataset" 
OUTPUT_BASE = "ROBUSTNESS_DATASETS"
SAMPLE_RATES = {
    "96k": {"sr": 96000, "window": 512,  "hop": 128},
    "48k": {"sr": 48000, "window": 256,  "hop": 64},
    "24k": {"sr": 24000, "window": 128,  "hop": 32}
}

# --- HELPER FUNCTIONS ---
def generate_spectrogram(audio, sr, window, hop, save_path):
    # 1. Compute STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=window, hop_length=hop)), ref=np.max)
    
    # 2. Draw Image
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(D, aspect='auto', origin='lower', cmap='viridis')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_file(wav_path, split_name):
    try:
        # 1. Check for Label
        label_path = wav_path.with_suffix(".txt")
        if not label_path.exists():
             label_path = wav_path.with_suffix(".TXT")
             if not label_path.exists():
                return

        # 2. Load Audio
        y, orig_sr = librosa.load(wav_path, sr=None)
        
        # 3. Process 3 Versions
        for name, config in SAMPLE_RATES.items():
            target_sr = config["sr"]
            window = config["window"]
            hop = config["hop"]
            
            # Create Folders
            base_dir = Path(OUTPUT_BASE) / name
            img_dir = base_dir / "images" / split_name
            lbl_dir = base_dir / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            # Downsample
            resample_ratio = target_sr / orig_sr
            num_samples = int(len(y) * resample_ratio)
            y_down = scipy.signal.resample(y, num_samples)

            # Save Image & Label
            save_name = wav_path.stem + ".png"
            generate_spectrogram(y_down, target_sr, window, hop, img_dir / save_name)
            shutil.copy(label_path, lbl_dir / (wav_path.stem + ".txt"))

        print(f"Processed: {wav_path.name}")

    except Exception as e:
        print(f"Error processing {wav_path.name}: {e}")

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("--- Starting Dataset Generation ---")
    
    # Find files
    all_wavs = list(Path(SOURCE_DIR).rglob("*.WAV")) + list(Path(SOURCE_DIR).rglob("*.wav"))
    all_wavs = list(set(all_wavs))
    print(f"Found {len(all_wavs)} audio files.")
    
    if len(all_wavs) == 0:
        print("ERROR: No WAV files found! Check your SOURCE_DIR.")
        exit()

    # Split Data
    random.shuffle(all_wavs)
    n = len(all_wavs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    splits = {
        "train": all_wavs[:train_end],
        "val":   all_wavs[train_end:val_end],
        "test":  all_wavs[val_end:]
    }

    # Run Loop
    for split_name, files in splits.items():
        print(f"--- Processing {len(files)} files for {split_name} set ---")
        for f in files:
            process_file(f, split_name)

    print("--- ALL DONE! ---")
