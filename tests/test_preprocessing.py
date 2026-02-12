import librosa
import numpy as np
import os
import asyncio
import soundfile as sf
from app.services.audio_processor import preprocess_audio
from app.services.audio_processor import pitch

audio_path = r"uploads\696cc14adcc1de89480eb579\20260118_111800.mp3"
async def main():
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return
    print("========== BEFORE ==========")
    y_before, sr_before = librosa.load(audio_path, sr=None)
    print(f"Sample rate: {sr_before}")
    print(f"Duration: {len(y_before) / sr_before:.2f}s")
    print(f"Max amplitude: {np.max(np.abs(y_before)):.4f}")
    # 2. Use await to actually execute the preprocessing
    y, sr = librosa.load(audio_path, sr=16000)
        
    # Voice Activity Detection (VAD)
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) > 0:
        y_voiced = np.concatenate([y[s:e] for s, e in intervals])
    else:
        y_voiced = y  # Fallback: keep original if no voice detected
        
    # Per-utterance normalization
    if np.max(np.abs(y_voiced)) > 0:
        y_voiced = y_voiced / np.max(np.abs(y_voiced))
        
    # Overwrite the file with processed audio
    sf.write(audio_path, y_voiced, 16000)
    print("Preprocessing completed successfully")
    print("\n========== AFTER ==========")
    y_after, sr_after = librosa.load(audio_path, sr=None)
    print(f"Sample rate: {sr_after}")
    print(f"Duration: {len(y_after) / sr_after:.2f}s")
    print(f"Max amplitude: {np.max(np.abs(y_after)):.4f}")
    
    print("\n========== BEFORE RAW DATA ==========")
    # Print the first 10 numerical samples
    print("Raw data (first 10 samples):", y_before[:10])

    # Print the last 10 numerical samples
    print("Raw data (last 10 samples):", y_before[-10:])

    print("\n========== AFTER RAW DATA ==========")
    # Print the first 10 numerical samples
    print("Raw data (first 10 samples):", y_after[:10])

    # Print the last 10 numerical samples
    print("Raw data (last 10 samples):", y_after[-10:])

    print(f"Total number of samples: {y_before.shape[0]}")
    print(f"Data numerical type: {y_before.dtype}") 

    print(f"Total number of samples: {y_after.shape[0]}")
    print(f"Data numerical type: {y_after.dtype}") 


    f0,voiced_flag=await pitch(y,sr)
    print(f"Pitch: {f0}")
    print(f"Voiced flag: {voiced_flag}")
    i=0
    for i in range(len(f0)):
        if voiced_flag[i]==True:
            print(f"Pitch: {f0[i]}")
            print(f"Voiced flag: {voiced_flag[i]}")
            i+=1
    print(f"Total number of voiced frames: {i}")



if __name__ == "__main__":
    asyncio.run(main())