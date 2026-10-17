import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Запуск на устройстве: {device}")

import os
from pathlib import Path

local_model_path = Path(__file__).resolve().parent / "model" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
model_id_or_path = str(local_model_path) if local_model_path.exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

print(f"Загружаем модель из: {model_id_or_path}...")
model = Qwen3TTSModel.from_pretrained(
    model_id_or_path,
    device_map=device,
    dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    attn_implementation="eager"  
)

print("Генерируем аудио...")
wavs, sr = model.generate_custom_voice(
    text="Артур! Здарова заебал, как ты че ты чем ты занят чепуха биля",
    language="Russian",
    speaker="Vivian", 
    instruct="Happy tone"
)

output_file = "result.wav"
sf.write(output_file, wavs[0], sr)
print(f"Готово! Слушайте файл {output_file}")