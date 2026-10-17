# -*- coding: utf-8 -*-
import os
import sys
import re
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
import gc
import logging
import warnings
from pathlib import Path
from datetime import datetime
import subprocess


# Подавление безобидной ошибки Windows asyncio WinError 10054 при закрытии/обновлении вкладки браузера
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_conn_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_conn_lost(self, exc):
            try:
                _orig_call_conn_lost(self, exc)
            except (ConnectionResetError, OSError):
                if hasattr(self, "_sock") and self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                if hasattr(self, "_server") and self._server is not None:
                    try:
                        self._server._detach(self)
                    except Exception:
                        pass
                    self._server = None
                self._called_connection_lost = True

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_conn_lost
    except Exception:
        pass

# Подавление предупреждений сторонних библиотек
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

import transformers.configuration_utils
transformers.configuration_utils.layer_type_validation = lambda *args, **kwargs: None
try:
    import qwen_tts.core.models.configuration_qwen3_tts as _qwen_cfg
    _qwen_cfg.layer_type_validation = lambda *args, **kwargs: None
except Exception:
    pass

import torch
import soundfile as sf
import gradio as gr

# Директории проекта
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "model"
GGUF_DIR = MODELS_DIR / "gguf"
GGUF_DIR.mkdir(exist_ok=True, parents=True)
BIN_DIR = BASE_DIR / "bin"
QWEN_TTS_EXE = BIN_DIR / "qwen-tts.exe"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_VARIANTS = {
    "1.7B - GGUF Q8_0 (~2.1 GB, Рекомендуется: быстро, высокое качество, поддержка эмоций)": {
        "desc": "8-битное квантование GGUF через нативный движок C++ с ускорением NVIDIA CUDA. Оптимальный баланс качества, скорости и поддержка эмоций.",
    },
    "1.7B - GGUF Q4_K_M (~1.2 GB, Быстрая: минимальный VRAM, поддержка эмоций)": {
        "desc": "4-битное квантование GGUF. Мгновенная генерация речи, пониженное потребление памяти при сохранении поддержки эмоций.",
    },
    "1.7B - GGUF BF16 (~3.9 GB, Без сжатия: студийная точность, поддержка эмоций)": {
        "desc": "16-битный формат без сжатия в C++ GGUF. Эталонное качество с полной поддержкой инструкций стиля и эмоций.",
    },
    "1.7B - PyTorch BF16 (~3.9 GB, Оригинал Hugging Face, поддержка эмоций)": {
        "desc": "Официальная модель Hugging Face через стандартную библиотеку PyTorch. Поддерживает эмоции.",
    },
    "0.6B - GGUF Q8_0 (~990 MB, Быстрая и легкая, без поддержки эмоций)": {
        "desc": "Облегченная модель 0.6B. Быстрый синтез, но архитектура 0.6B не поддерживает эмоции.",
    },
    "0.6B - GGUF Q4_K_M (~630 MB, Ультралегкая для слабых GPU, без поддержки эмоций)": {
        "desc": "Минимальный размер (всего ~630 МБ) для слабых систем или ультра-быстрого синтеза. Эмоции не поддерживаются.",
    },
    "0.6B - GGUF BF16 (~1.8 GB, 0.6B без сжатия, без поддержки эмоций)": {
        "desc": "Компактная 0.6B модель в точности BF16 без квантования. Эмоции не поддерживаются.",
    },
    "0.6B - PyTorch BF16 (~1.4 GB, Оригинал Hugging Face 0.6B, без поддержки эмоций)": {
        "desc": "Официальная 0.6B модель Hugging Face через PyTorch. Эмоции не поддерживаются.",
    },
}

DEFAULT_MODEL = "1.7B - GGUF Q8_0 (~2.1 GB, Рекомендуется: быстро, высокое качество, поддержка эмоций)"
MODEL_CHOICES_CUSTOM = list(MODEL_VARIANTS.keys())
MODEL_CHOICES_CLONE = list(MODEL_VARIANTS.keys())

AVAILABLE_MODELS = {
    ("CustomVoice", "1.7B"): {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "local_dir": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "name": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "desc": "1.7B параметров — высокое качество, поддержка дикторов и инструкций/эмоций",
    },
    ("CustomVoice", "0.6B"): {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "local_dir": MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "name": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "desc": "0.6B параметров — высокая скорость, экономия VRAM (без поддержки эмоций)",
    },
    ("Base", "1.7B"): {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
        "name": "Qwen3-TTS-12Hz-1.7B-Base",
        "desc": "1.7B параметров — высокое качество клонирования голоса",
    },
    ("Base", "0.6B"): {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "local_dir": MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
        "name": "Qwen3-TTS-12Hz-0.6B-Base",
        "desc": "0.6B параметров — быстрое клонирование голоса с низким расходом VRAM",
    },
}

# Определение устройства
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

from qwen_tts import Qwen3TTSModel

# --- ГЛОБАЛЬНЫЙ КЭШ МОДЕЛИ ---
loaded_model_key = None
loaded_model = None

def parse_model_size(size_str: str) -> str:
    if size_str and "0.6" in str(size_str):
        return "0.6B"
    return "1.7B"

def parse_model_selection(label: str, mode: str):
    """
    Разбирает выбор модели из выпадающего списка.
    mode: 'CustomVoice' или 'Base'
    """
    label_lower = (label or "").lower()
    is_06b = "0.6" in label_lower
    is_gguf = "gguf" in label_lower

    if is_gguf:
        size_str = "0.6b" if is_06b else "1.7b"
        mode_str = "customvoice" if mode == "CustomVoice" else "base"
        if "q4" in label_lower:
            quant = "Q4_K_M"
        elif "bf16" in label_lower:
            quant = "BF16"
        else:
            quant = "Q8_0"

        talker_file = f"qwen-talker-{size_str}-{mode_str}-{quant}.gguf"
        codec_file = f"qwen-tokenizer-12hz-{quant}.gguf"
        return {
            "backend": "gguf",
            "size": "0.6B" if is_06b else "1.7B",
            "quant": quant,
            "talker_file": talker_file,
            "codec_file": codec_file,
            "name": f"Qwen3-TTS-{size_str.upper()}-{mode} [GGUF {quant}]",
        }
    else:
        size = "0.6B" if is_06b else "1.7B"
        return {
            "backend": "pytorch",
            "size": size,
            "name": AVAILABLE_MODELS.get((mode, size), {}).get("name", f"Qwen3-TTS-{size}-{mode} [PyTorch BF16]"),
        }

def free_pytorch_vram():
    global loaded_model, loaded_model_key
    if loaded_model is not None:
        print("[+] Освобождение PyTorch VRAM перед запуском GGUF...")
        del loaded_model
        loaded_model = None
        loaded_model_key = None
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

def ensure_gguf_file(filename: str, progress_fn=None) -> Path:
    target_path = GGUF_DIR / filename
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    if progress_fn:
        progress_fn(0.1, desc=f"Скачивание {filename} с Hugging Face...")
    print(f"[+] Скачивание GGUF модели '{filename}' с Hugging Face (Serveurperso/Qwen3-TTS-GGUF)...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="Serveurperso/Qwen3-TTS-GGUF",
        filename=filename,
        local_dir=str(GGUF_DIR),
    )
    if progress_fn:
        progress_fn(0.4, desc=f"Файл {filename} загружен!")
    return Path(path)

def run_gguf_inference(
    mode: str,
    model_cfg: dict,
    text: str,
    speaker: str = "Vivian",
    language: str = "Russian",
    instruct: str = None,
    ref_audio: str = None,
    ref_text: str = None,
    x_vector_mode: bool = False,
    limit_tokens: int = 300,
    temperature: float = 0.9,
    top_p: float = 1.0,
    top_k: int = 50,
    progress_fn = None,
):
    if not QWEN_TTS_EXE.exists():
        raise FileNotFoundError(
            f"Исполняемый файл GGUF движка не найден: {QWEN_TTS_EXE}.\n"
            "Убедитесь, что qwen-tts.exe и необходимые библиотеки находятся в папке bin/ проекта."
        )

    if progress_fn:
        progress_fn(0.05, desc=f"Подготовка модели {model_cfg['name']}...")

    # Проверяем / скачиваем веса Talker и Codec
    talker_path = ensure_gguf_file(model_cfg["talker_file"], progress_fn=progress_fn)
    codec_path = ensure_gguf_file(model_cfg["codec_file"], progress_fn=progress_fn)

    # Освобождаем память PyTorch, чтобы GPU был полностью свободен для GGUF движка
    free_pytorch_vram()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"custom_gguf_{model_cfg['size']}_{model_cfg['quant']}_{speaker}" if mode == "CustomVoice" else f"clone_gguf_{model_cfg['size']}_{model_cfg['quant']}"
    out_filename = f"{prefix}_{timestamp}.wav"
    output_file = OUTPUTS_DIR / out_filename

    cmd = [
        str(QWEN_TTS_EXE),
        "--model", str(talker_path),
        "--codec", str(codec_path),
        "-o", str(output_file),
    ]

    temp_ref_txt = None
    try:
        if mode == "CustomVoice":
            cmd.extend(["--speaker", speaker])
            if instruct and instruct.strip():
                cmd.extend(["--instruct", instruct.strip()])
        else: # Base (Voice Clone)
            if not ref_audio:
                raise ValueError("Не указан аудио-образец для клонирования голоса.")
            cmd.extend(["--ref-wav", str(ref_audio)])
            if ref_text and not x_vector_mode and ref_text.strip():
                temp_ref_txt = OUTPUTS_DIR / f"_ref_{timestamp}.txt"
                temp_ref_txt.write_text(ref_text.strip(), encoding="utf-8")
                cmd.extend(["--ref-text", str(temp_ref_txt)])

        if language and language != "Auto":
            cmd.extend(["--lang", language])
        if limit_tokens:
            cmd.extend(["--max-new", str(int(limit_tokens))])
        if temperature is not None:
            cmd.extend(["--temp", str(float(temperature))])
        if top_p is not None:
            cmd.extend(["--top-p", str(float(top_p))])
        if top_k is not None:
            cmd.extend(["--top-k", str(int(top_k))])

        proc_env = dict(os.environ)
        proc_env["PATH"] = str(BIN_DIR) + os.pathsep + proc_env.get("PATH", "")

        if progress_fn:
            progress_fn(0.2, desc="Генерация речи в C++ GGUF движке...")

        t_start = time.time()
        res = subprocess.run(
            cmd,
            input=text.strip().encode("utf-8"),
            capture_output=True,
            cwd=str(BASE_DIR),
            env=proc_env,
        )
        gen_time = time.time() - t_start

        if res.returncode != 0:
            err_msg = res.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"GGUF движок завершился с кодом {res.returncode}:\n{err_msg}")

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise RuntimeError("Выходной аудиофайл не был создан движком.")

        info = sf.info(str(output_file))
        audio_dur = info.duration
        rtf = gen_time / audio_dur if audio_dur > 0 else 0

        return str(output_file), out_filename, gen_time, audio_dur, rtf

    finally:
        if temp_ref_txt and temp_ref_txt.exists():
            try:
                temp_ref_txt.unlink()
            except Exception:
                pass

def get_vram_stats():
    if DEVICE == "cpu":
        return "Режим CPU"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return f"VRAM: {allocated:.0f} MB занято / {reserved:.0f} MB зарезервировано (Пик: {max_alloc:.0f} MB)"

def load_engine(model_type="CustomVoice", model_size="1.7B"):
    global loaded_model_key, loaded_model
    model_size = parse_model_size(model_size)
    target_key = (model_type, model_size)

    if loaded_model is not None and loaded_model_key == target_key:
        return loaded_model

    info = AVAILABLE_MODELS.get(target_key)
    if not info:
        raise ValueError(f"Неизвестная конфигурация модели: {target_key}")

    local_path = info["local_dir"]
    repo_id = info["repo_id"]

    # Проверка наличия локальной модели
    if local_path.exists() and (local_path / "config.json").exists():
        target_path = str(local_path)
        print(f"[+] Используется локальная папка: {target_path}")
    else:
        print(f"[+] Модель {repo_id} не найдена в {local_path}.")
        print(f"[+] Автоматическое скачивание модели {repo_id} в {local_path}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
            )
            target_path = str(local_path)
            print(f"[+] Модель успешно скачана в {target_path}")
        except Exception as dl_err:
            print(f"[!] Не удалось скачать в локальную папку: {dl_err}. Попытка прямой загрузки из Hugging Face...")
            target_path = repo_id

    print(f"[+] Загрузка весов модели: {model_type} ({model_size})...")
    if loaded_model is not None:
        del loaded_model
        loaded_model = None
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        target_path,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="eager",
    )
    if hasattr(model.model, "talker"):
        model.model.talker.config._attn_implementation = "eager"
        if hasattr(model.model.talker, "model"):
            model.model.talker.model.config._attn_implementation = "eager"

    loaded_model = model
    loaded_model_key = target_key
    print(f"[+] Модель {model_type} ({model_size}) успешно загружена за {time.time() - t0:.2f} сек. Путь: {target_path}")
    return loaded_model


# --- СПИСОК ДИКТОРОВ И ИХ ОПИСАНИЯ ---
SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

SPEAKER_INFO = {
    "Vivian": {
        "gender": "Женский (Female)",
        "native": "Китайский",
        "desc": "Яркий, выразительный молодой женский голос. Универсален для любых эмоций и языков."
    },
    "Serena": {
        "gender": "Женский (Female)",
        "native": "Китайский",
        "desc": "Мягкий, тёплый, дружелюбный и нежный тембр."
    },
    "Uncle_Fu": {
        "gender": "Мужской (Male)",
        "native": "Китайский",
        "desc": "Зрелый, глубокий, бархатистый голос рассказчика (аудиокниги, повествования)."
    },
    "Dylan": {
        "gender": "Мужской (Male)",
        "native": "Китайский (Пекин)",
        "desc": "Молодой, ясный, естественный мужской голос с чистой дикцией."
    },
    "Eric": {
        "gender": "Мужской (Male)",
        "native": "Китайский (Сычуань)",
        "desc": "Живой, бодрый мужской голос с легкой хрипотцой (сычуаньский стиль)."
    },
    "Ryan": {
        "gender": "Мужской (Male)",
        "native": "Английский",
        "desc": "Динамичный мужской голос с сильным ритмом и напором."
    },
    "Aiden": {
        "gender": "Мужской (Male)",
        "native": "Английский",
        "desc": "Солнечный, уверенный американский мужской голос с чистым средним регистром."
    },
    "Ono_Anna": {
        "gender": "Женский (Female)",
        "native": "Японский",
        "desc": "Игривый, звонкий и легкий японский женский голос (аниме/влог)."
    },
    "Sohee": {
        "gender": "Женский (Female)",
        "native": "Корейский",
        "desc": "Тёплый, эмоциональный корейский женский голос с богатой интонацией."
    },
}

LANGUAGES = [
    "Auto",
    "Russian",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Italian",
]

def calculate_smart_tokens(text: str) -> int:
    chars = len(text.strip()) if text else 10
    # При 12 Гц 1 секунда аудио = 12 токенов.
    # В естественной русской/английской речи темп ~10-14 символов в сек (~1 токен/символ).
    # Запас 1.5x на паузы, пунктуацию и интонацию + 60 токенов базового запаса:
    return max(60, min(1500, int(chars * 1.5) + 60))

# --- АДАПТЕР ЭМОЦИЙ И ИНСТРУКЦИЙ ДЛЯ МОДЕЛИ QWEN3-TTS ---
# Модель Qwen3-TTS обучалась на английских (и китайских) промптах в качестве instruction-tuning.
# На русские фразы вроде "радостным голосом" нейросеть слабо реагирует, поэтому мы автоматически
# преобразуем русские эмоциональные описания в соответствующие обученные английские промпты.
EMOTION_RULES = [
    (r"(радост|весел|улыбк|счастлив|позитив|радость|сме[её]|празднич)", "Speak in a very happy, cheerful and joyful tone with a bright smile"),
    (r"(шепот|ш[её]пт|тихо|по секрету|приглуш)", "Whisper quietly in a breathy and mysterious low voice"),
    (r"(грустн|печал|тоск|плач|сочувств|депресс|жалоб|слез|гореч)", "Speak in an extremely sad, sorrowful, quiet and depressed tone with slow pace"),
    (r"(зл[оые]|сердит|ярост|раздраж|крич|агрессив|гнев|беш[её]н)", "Speak in an intensely angry, furious, aggressive and shouting tone"),
    (r"(спокойн|размерен|мирн|невозмут|релакс|умиротворен)", "Speak in a very calm, relaxed, peaceful and gentle tone, speaking slowly and softly"),
    (r"(соблазн|сексуальн|чувственн|томн)", "Speak in an attractive, seductive, warm, captivating and sultry voice"),
    (r"(диктор|новост|официальн|строг|серь[её]зн|доклад)", "Speak in a formal, professional news broadcast anchor tone, serious and authoritative"),
    (r"(рассказчик|повествова|сказк|истори|аудиокниг)", "Speak in an engaging, expressive storytelling narrative tone"),
    (r"(уверен|тверд|решительн|убедительн)", "Speak in a confident, firm, clear and assertive tone"),
    (r"(восторжен|восхищ|вдохновен|эйфори)", "Speak in an enthusiastic, amazed, inspiring and euphoric tone"),
    (r"(сарказм|ирони|насмешк|язвит)", "Speak in a sarcastic, mocking, ironic and humorous tone"),
    (r"(таинственн|загадочн|мистик|страшн|ужас|пугающ|зловещ)", "Speak in a mysterious, dark, eerie and suspenseful narrator voice"),
    (r"(быстр|динамич|энергич|тороп|скорост)", "Speak very fast at a rapid, energetic tempo without pauses"),
    (r"(медленн|затяж|тягуч|растягива)", "Speak very slowly with long pauses between words, relaxed pace"),
    (r"(ласков|нежн|с любовью|мило|мил[ыйая])", "Speak in a gentle, tender, sweet, loving and warm voice"),
    (r"(устал|утомлен|изможд|вял|сонн)", "Speak in an exhausted, tired, weary, sluggish and faint voice"),
    (r"(тревожн|беспокойн|нервн|паник|напряжен|страх)", "Speak in an anxious, nervous, tense, worried and hurried tone"),
]


def format_instruction_for_model(raw_instruct: str):
    """
    Преобразует пользовательскую инструкцию в понятный для Qwen3-TTS формат.
    Возвращает (translated_instruct, display_note).
    """
    if not raw_instruct or not raw_instruct.strip():
        return None, "Стандартный (нейтральный)"

    cleaned = raw_instruct.strip()
    matched = []
    for pattern, en_prompt in EMOTION_RULES:
        if re.search(pattern, cleaned, re.IGNORECASE):
            matched.append(en_prompt)

    if matched:
        final_instruct = ", ".join(matched)
        display_note = f'"{cleaned}" -> [Оптимизировано: {final_instruct}]'
        return final_instruct, display_note
    else:
        # Если инструкция уже на английском или специфичная
        return cleaned, f'"{cleaned}"'

# --- ОБРАБОТЧИКИ ГЕНЕРАЦИИ ---
def synthesize_custom_voice(text, speaker, language, instruct, model_size_label, auto_tokens, max_tokens, temperature, top_p, top_k, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Ошибка: Введите текст для озвучки."

    try:
        model_cfg = parse_model_selection(model_size_label, mode="CustomVoice")
        limit_tokens = calculate_smart_tokens(text) if auto_tokens else int(max_tokens)
        lang_arg = None if (language == "Auto" or not language) else language
        final_instruct, instruct_note = format_instruction_for_model(instruct)

        instruct_warning = ""
        if model_cfg["size"] == "0.6B" and final_instruct and model_cfg["backend"] == "pytorch":
            instruct_warning = "\nПримечание: Модель 0.6B PyTorch не поддерживает параметр instruct (эмоции/темп игнорируются, используйте 1.7B для эмоций)."

        if model_cfg["backend"] == "gguf":
            output_file, filename, gen_time, audio_dur, rtf = run_gguf_inference(
                mode="CustomVoice",
                model_cfg=model_cfg,
                text=text.strip(),
                speaker=speaker,
                language=lang_arg,
                instruct=final_instruct,
                limit_tokens=limit_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                progress_fn=progress,
            )
            engine_info = f"Native C++ GGUF ({model_cfg['quant']}) [CUDA sm_120 / RTX 5080]"
        else:
            if DEVICE != "cpu":
                torch.cuda.reset_peak_memory_stats()

            if progress:
                progress(0.1, desc=f"Загрузка PyTorch модели {model_cfg['size']}...")
            model = load_engine("CustomVoice", model_cfg["size"], progress_fn=progress)

            if progress:
                progress(0.3, desc="Синтез аудио в PyTorch...")
            t_start = time.time()
            wavs, sr = model.generate_custom_voice(
                text=text.strip(),
                speaker=speaker,
                language=lang_arg,
                instruct=final_instruct,
                max_new_tokens=limit_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            gen_time = time.time() - t_start
            audio_dur = len(wavs[0]) / sr
            rtf = gen_time / audio_dur if audio_dur > 0 else 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_pytorch_{model_cfg['size']}_{speaker}_{timestamp}.wav"
            output_file = str(OUTPUTS_DIR / filename)
            sf.write(output_file, wavs[0], sr)
            engine_info = f"PyTorch ({DTYPE}) | {get_vram_stats()}"

        log = (
            f"Статус: Успешно синтезировано\n"
            f"Модель: {model_cfg['name']}\n"
            f"Движок: {engine_info}\n"
            f"Диктор: {speaker} ({SPEAKER_INFO.get(speaker, {}).get('gender', '')})\n"
            f"Инструкция: {instruct_note}{instruct_warning}\n"
            f"Язык: {language}\n"
            f"Время генерации: {gen_time:.2f} сек\n"
            f"Длительность аудио: {audio_dur:.2f} сек (RTF: {rtf:.2f}x)\n"
            f"Файл: outputs/{filename}"
        )
        return str(output_file), log

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Ошибка генерации: {str(e)}"

def synthesize_voice_clone(text, ref_audio, ref_text, x_vector_mode, language, model_size_label, auto_tokens, max_tokens, temperature, top_p, top_k, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Ошибка: Введите текст для синтеза."
    if not ref_audio:
        return None, "Ошибка: Загрузите или запишите аудио-образец голоса."
    if not x_vector_mode and (not ref_text or not ref_text.strip()):
        return None, "Ошибка: Укажите текст из аудио-образца либо включите режим 'Только тембр'."

    try:
        model_cfg = parse_model_selection(model_size_label, mode="Base")
        limit_tokens = calculate_smart_tokens(text) if auto_tokens else int(max_tokens)
        lang_arg = None if (language == "Auto" or not language) else language

        if model_cfg["backend"] == "gguf":
            output_file, filename, gen_time, audio_dur, rtf = run_gguf_inference(
                mode="Base",
                model_cfg=model_cfg,
                text=text.strip(),
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_mode=x_vector_mode,
                language=lang_arg,
                limit_tokens=limit_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                progress_fn=progress,
            )
            engine_info = f"Native C++ GGUF ({model_cfg['quant']}) [CUDA sm_120 / RTX 5080]"
        else:
            if DEVICE != "cpu":
                torch.cuda.reset_peak_memory_stats()

            if progress:
                progress(0.1, desc=f"Загрузка PyTorch Base модели {model_cfg['size']}...")
            model = load_engine("Base", model_cfg["size"], progress_fn=progress)

            if progress:
                progress(0.3, desc="Клонирование голоса в PyTorch...")
            t_start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                ref_audio=ref_audio,
                ref_text=ref_text.strip() if (ref_text and not x_vector_mode) else None,
                x_vector_only_mode=bool(x_vector_mode),
                language=lang_arg,
                max_new_tokens=limit_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            gen_time = time.time() - t_start
            audio_dur = len(wavs[0]) / sr
            rtf = gen_time / audio_dur if audio_dur > 0 else 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clone_pytorch_{model_cfg['size']}_{timestamp}.wav"
            output_file = str(OUTPUTS_DIR / filename)
            sf.write(output_file, wavs[0], sr)
            engine_info = f"PyTorch ({DTYPE}) | {get_vram_stats()}"

        mode_str = "X-Vector (только тембр)" if x_vector_mode else "Full ICL (с текстом образца)"
        log = (
            f"Статус: Голос успешно клонирован\n"
            f"Модель: {model_cfg['name']}\n"
            f"Движок: {engine_info}\n"
            f"Режим: {mode_str}\n"
            f"Язык: {language}\n"
            f"Время синтеза: {gen_time:.2f} сек\n"
            f"Длительность аудио: {audio_dur:.2f} сек (RTF: {rtf:.2f}x)\n"
            f"Файл: outputs/{filename}"
        )
        return str(output_file), log

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Ошибка клонирования: {str(e)}"

# Описание диктора
def update_speaker_badge(speaker_name):
    info = SPEAKER_INFO.get(speaker_name, {})
    gender = info.get("gender", "")
    native = info.get("native", "")
    desc = info.get("desc", "")
    return f"**Описание диктора:** {gender}, родной язык {native.lower()}. {desc}"

def filter_speakers_by_gender(gender_choice, current_speaker):
    if gender_choice == "Женский":
        filtered = [s for s in SPEAKERS if "Женский" in SPEAKER_INFO.get(s, {}).get("gender", "")]
    elif gender_choice == "Мужской":
        filtered = [s for s in SPEAKERS if "Мужской" in SPEAKER_INFO.get(s, {}).get("gender", "")]
    else:
        filtered = list(SPEAKERS)
    
    new_speaker = current_speaker if current_speaker in filtered else filtered[0]
    new_badge = update_speaker_badge(new_speaker)
    return gr.update(choices=filtered, value=new_speaker), new_badge


# --- ИНТЕРФЕЙС GRADIO ---
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

custom_css = """
.gradio-container,
.gradio-container.contain,
.gradio-container.fill-width {
    width: 95% !important;
    max-width: 1800px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
    box-sizing: border-box !important;
}
.quick-btn {
    min-width: auto !important;
    padding: 6px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
.quick-btn-clear {
    min-width: auto !important;
    padding: 6px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
"""

with gr.Blocks(title="Qwen3-TTS WebUI", css=custom_css, theme=theme, fill_width=True) as demo:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        cuda_info = f"CUDA sm_{cap[0]}{cap[1]}"
    else:
        gpu_name = "CPU"
        cuda_info = "CPU"

    gguf_status = f"Native C++ GGUF ({cuda_info})" if QWEN_TTS_EXE.exists() else "Native C++ GGUF (не найден)"
    engines_str = f"{gguf_status} и PyTorch"

    gr.Markdown(
        f"""
        # Qwen3-TTS WebUI
        ### Синтез речи и клонирование голоса (10+ языков, включая русский)
        **Устройство:** `{DEVICE}` ({gpu_name}) &nbsp;|&nbsp; **Движки:** {engines_str}
        """
    )

    model_dropdown = gr.Dropdown(
        label="Выбор модели (GGUF квантование или PyTorch)",
        choices=list(MODEL_VARIANTS.keys()),
        value=DEFAULT_MODEL,
        interactive=True,
    )

    with gr.Tabs() as tabs:
        # ------------------- ВКЛАДКА 1: CUSTOM VOICE -------------------
        with gr.TabItem("Синтез речи (Custom Voice)", id="tab_custom"):
            with gr.Row():
                # ЛЕВАЯ КОЛОНКА (scale=3)
                with gr.Column(scale=3):
                    cv_text = gr.Textbox(
                        label="Текст для синтеза речи",
                        placeholder="Введите текст для озвучки...",
                        value="Привет! Это проверка синтеза речи в системе Qwen3-TTS с поддержкой естественных интонаций и эмоций.",
                        lines=7,
                    )

                    cv_instruct = gr.Textbox(
                        label="Инструкция к интонации, эмоциям и темпу (Instruct)",
                        placeholder="Например: Speak in a very calm, relaxed, peaceful and gentle tone, speaking slowly and softly",
                        value="",
                        lines=2,
                        info="Модели 1.7B поддерживают прямое управление интонацией через естественные фразы. Ниже доступны быстрые пресеты."
                    )

                    PRESET_PROMPTS = {
                        "Спокойно": "Speak in a very calm, relaxed, peaceful and gentle tone, speaking slowly and softly",
                        "Радостно": "Speak in a very happy, cheerful and joyful tone with a bright smile",
                        "Грустно": "Speak in an extremely sad, sorrowful, quiet and depressed tone with slow pace",
                        "Сердито": "Speak in an intensely angry, furious, aggressive and shouting tone",
                        "Шёпот": "Whisper quietly in a breathy and mysterious low voice",
                        "Соблазнительно": "Speak in an attractive, seductive, warm, captivating and sultry voice",

                        "Диктор": "Speak in a formal, professional news broadcast anchor tone, serious and authoritative",
                        "Рассказчик": "Speak in an engaging, expressive storytelling narrative tone",
                        "Уверенно": "Speak in a confident, firm, clear and assertive tone",
                        "Восторженно": "Speak in an enthusiastic, amazed, inspiring and euphoric tone",
                        "Иронично": "Speak in a sarcastic, mocking, ironic and humorous tone",
                        "Загадочно": "Speak in a mysterious, dark, eerie and suspenseful narrator voice",

                        "Быстро": "Speak very fast at a rapid, energetic tempo without pauses",
                        "Медленно": "Speak very slowly with long pauses between words, relaxed pace",
                        "Нежно": "Speak in a gentle, tender, sweet, loving and warm voice",
                        "Устало": "Speak in an exhausted, tired, weary, sluggish and faint voice",
                        "Тревожно": "Speak in an anxious, nervous, tense, worried and hurried tone",
                    }

                    with gr.Row():
                        btn_calm = gr.Button("Спокойно", size="sm", elem_classes=["quick-btn"])
                        btn_joy = gr.Button("Радостно", size="sm", elem_classes=["quick-btn"])
                        btn_sad = gr.Button("Грустно", size="sm", elem_classes=["quick-btn"])
                        btn_angry = gr.Button("Сердито", size="sm", elem_classes=["quick-btn"])
                        btn_whisper = gr.Button("Шёпот", size="sm", elem_classes=["quick-btn"])
                        btn_seductive = gr.Button("Соблазнительно", size="sm", elem_classes=["quick-btn"])

                    with gr.Row():
                        btn_anchor = gr.Button("Диктор", size="sm", elem_classes=["quick-btn"])
                        btn_story = gr.Button("Рассказчик", size="sm", elem_classes=["quick-btn"])
                        btn_confident = gr.Button("Уверенно", size="sm", elem_classes=["quick-btn"])
                        btn_enthusiastic = gr.Button("Восторженно", size="sm", elem_classes=["quick-btn"])
                        btn_ironic = gr.Button("Иронично", size="sm", elem_classes=["quick-btn"])
                        btn_mysterious = gr.Button("Загадочно", size="sm", elem_classes=["quick-btn"])

                    with gr.Row():
                        btn_fast = gr.Button("Быстро", size="sm", elem_classes=["quick-btn"])
                        btn_slow = gr.Button("Медленно", size="sm", elem_classes=["quick-btn"])
                        btn_tender = gr.Button("Нежно", size="sm", elem_classes=["quick-btn"])
                        btn_tired = gr.Button("Устало", size="sm", elem_classes=["quick-btn"])
                        btn_anxious = gr.Button("Тревожно", size="sm", elem_classes=["quick-btn"])
                        btn_clear_inst = gr.Button("Сбросить", size="sm", elem_classes=["quick-btn-clear"])

                    btn_calm.click(lambda: PRESET_PROMPTS["Спокойно"], outputs=[cv_instruct])
                    btn_joy.click(lambda: PRESET_PROMPTS["Радостно"], outputs=[cv_instruct])
                    btn_sad.click(lambda: PRESET_PROMPTS["Грустно"], outputs=[cv_instruct])
                    btn_angry.click(lambda: PRESET_PROMPTS["Сердито"], outputs=[cv_instruct])
                    btn_whisper.click(lambda: PRESET_PROMPTS["Шёпот"], outputs=[cv_instruct])
                    btn_seductive.click(lambda: PRESET_PROMPTS["Соблазнительно"], outputs=[cv_instruct])

                    btn_anchor.click(lambda: PRESET_PROMPTS["Диктор"], outputs=[cv_instruct])
                    btn_story.click(lambda: PRESET_PROMPTS["Рассказчик"], outputs=[cv_instruct])
                    btn_confident.click(lambda: PRESET_PROMPTS["Уверенно"], outputs=[cv_instruct])
                    btn_enthusiastic.click(lambda: PRESET_PROMPTS["Восторженно"], outputs=[cv_instruct])
                    btn_ironic.click(lambda: PRESET_PROMPTS["Иронично"], outputs=[cv_instruct])
                    btn_mysterious.click(lambda: PRESET_PROMPTS["Загадочно"], outputs=[cv_instruct])

                    btn_fast.click(lambda: PRESET_PROMPTS["Быстро"], outputs=[cv_instruct])
                    btn_slow.click(lambda: PRESET_PROMPTS["Медленно"], outputs=[cv_instruct])
                    btn_tender.click(lambda: PRESET_PROMPTS["Нежно"], outputs=[cv_instruct])
                    btn_tired.click(lambda: PRESET_PROMPTS["Устало"], outputs=[cv_instruct])
                    btn_anxious.click(lambda: PRESET_PROMPTS["Тревожно"], outputs=[cv_instruct])
                    btn_clear_inst.click(lambda: "", outputs=[cv_instruct])

                    with gr.Row():
                        cv_gender = gr.Dropdown(
                            choices=["Все", "Женский", "Мужской"],
                            value="Все",
                            label="Пол",
                            scale=1,
                        )
                        cv_speaker = gr.Dropdown(
                            choices=SPEAKERS,
                            value="Vivian",
                            label="Диктор",
                            scale=1,
                        )
                        cv_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="Russian",
                            label="Язык синтеза",
                            scale=1,
                        )

                    cv_speaker_info = gr.Markdown(value=update_speaker_badge("Vivian"))
                    cv_speaker.change(update_speaker_badge, inputs=[cv_speaker], outputs=[cv_speaker_info])
                    cv_gender.change(filter_speakers_by_gender, inputs=[cv_gender, cv_speaker], outputs=[cv_speaker, cv_speaker_info])

                    with gr.Accordion("Настройки генерации", open=False):
                        cv_auto_tokens = gr.Checkbox(
                            label="Автоматический расчет длины аудио",
                            value=True,
                            info="Рассчитывает количество аудио-токенов исходя из объема текста."
                        )
                        cv_max_tokens = gr.Slider(
                            minimum=50, maximum=1500, value=300, step=25,
                            label="Максимальное число токенов (Max New Tokens)",
                            interactive=True,
                        )
                        with gr.Row():
                            cv_temp = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature (Вариативность)")
                            cv_top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-P")
                            cv_top_k = gr.Slider(0, 100, value=50, step=5, label="Top-K")

                    cv_generate_btn = gr.Button("Синтезировать речь", variant="primary", size="lg")

                # ПРАВАЯ КОЛОНКА (scale=2)
                with gr.Column(scale=2):
                    cv_audio_out = gr.Audio(
                        label="Результат синтеза",
                        type="filepath",
                        interactive=False,
                        autoplay=True
                    )
                    cv_log_out = gr.Textbox(
                        label="Статус генерации",
                        lines=8,
                        interactive=False,
                    )

            cv_generate_btn.click(
                fn=synthesize_custom_voice,
                inputs=[cv_text, cv_speaker, cv_language, cv_instruct, model_dropdown, cv_auto_tokens, cv_max_tokens, cv_temp, cv_top_p, cv_top_k],
                outputs=[cv_audio_out, cv_log_out],
            )

            # Готовые примеры
            with gr.Accordion("Примеры для быстрого тестирования", open=False):
                gr.Examples(
                    examples=[
                        [
                            "Добро пожаловать в мир высококачественного синтеза речи Qwen3-TTS! Как я могу вам помочь сегодня?",
                            "Vivian",
                            "Russian",
                            "Speak in a very happy, cheerful and joyful tone with a bright smile",
                        ],
                        [
                            "Слушай внимательно... В старом заброшенном замке на холме каждую полночь зажигается таинственный огонек.",
                            "Uncle_Fu",
                            "Russian",
                            "Speak in a mysterious, dark, eerie and suspenseful narrator voice",
                        ],
                        [
                            "Внимание! Это экстренное сообщение. Просьба всем сохранять спокойствие и следовать инструкциям.",
                            "Ryan",
                            "Russian",
                            "Speak in a formal, professional news broadcast anchor tone, serious and authoritative",
                        ],
                        [
                            "The quick brown fox jumps over the lazy dog. Today is a great day to explore generative speech modeling!",
                            "Aiden",
                            "English",
                            "Speak very fast at a rapid, energetic tempo without pauses",
                        ],
                        [
                            "こんにちは！今日はとてもいい天気ですね。一緒にお散歩に行きませんか？",
                            "Ono_Anna",
                            "Japanese",
                            "Speak in a playful, cheerful and cute friendly voice",
                        ],
                    ],
                    inputs=[cv_text, cv_speaker, cv_language, cv_instruct],
                    label="Нажмите на любой пример для автоматической подстановки:",
                )

        # ------------------- ВКЛАДКА 2: КЛОНИРОВАНИЕ ГОЛОСА -------------------
        with gr.TabItem("Клонирование голоса (Voice Clone)", id="tab_clone"):
            with gr.Row():
                # ЛЕВАЯ КОЛОНКА (scale=3)
                with gr.Column(scale=3):
                    vc_target_text = gr.Textbox(
                        label="Новый текст для произнесения клонированным голосом",
                        placeholder="Введите текст, который должен сказать клонированный голос...",
                        value="Привет, теперь я говорю этим голосом! Это технология клонирования Qwen3-TTS.",
                        lines=7,
                    )

                    vc_audio_ref = gr.Audio(
                        label="Аудиозапись голоса (от 3 секунд чистой речи)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    vc_ref_text = gr.Textbox(
                        label="Текст из аудио-образца (транскрипт)",
                        placeholder="Напишите точный текст, который звучит в аудиозаписи выше (повышает качество клонирования)",
                        lines=2,
                    )

                    vc_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Russian",
                        label="Язык синтеза",
                    )

                    with gr.Accordion("Настройки генерации", open=False):
                        vc_xvec_mode = gr.Checkbox(
                            label="Режим тембра (без транскрипта образца)",
                            value=False,
                            info="Клонирует акустический тембр голоса без необходимости указывать точный текст образца.",
                        )
                        vc_auto_tokens = gr.Checkbox(label="Автоматический расчет токенов", value=True)
                        vc_max_tokens = gr.Slider(50, 1500, value=300, step=25, label="Max New Tokens")
                        with gr.Row():
                            vc_temp = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
                            vc_top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-P")
                            vc_top_k = gr.Slider(0, 100, value=50, step=5, label="Top-K")

                    vc_clone_btn = gr.Button("Клонировать голос и озвучить", variant="primary", size="lg")

                # ПРАВАЯ КОЛОНКА (scale=2)
                with gr.Column(scale=2):
                    vc_audio_out = gr.Audio(
                        label="Результат клонирования",
                        type="filepath",
                        interactive=False,
                        autoplay=True
                    )
                    vc_log_out = gr.Textbox(
                        label="Статус генерации",
                        lines=8,
                        interactive=False,
                    )

            vc_clone_btn.click(
                fn=synthesize_voice_clone,
                inputs=[vc_target_text, vc_audio_ref, vc_ref_text, vc_xvec_mode, vc_language, model_dropdown, vc_auto_tokens, vc_max_tokens, vc_temp, vc_top_p, vc_top_k],
                outputs=[vc_audio_out, vc_log_out],
            )

        # ------------------- ВКЛАДКА 3: СПРАВКА -------------------
        with gr.TabItem("Справка и документация", id="tab_info"):
            gr.Markdown(
                """
                ### Таблица доступных дикторов (CustomVoice)
                Все дикторы могут говорить на любом из поддерживаемых языков (включая русский). Для максимальной естественности рекомендуется учитывать их базовый профиль:

                | Диктор | Пол | Родной язык | Характеристика тембра |
                |---|---|---|---|
                | **Vivian** | Женский | Китайский | Яркий, чистый, молодой женский голос, отличная дикция |
                | **Serena** | Женский | Китайский | Мягкий, теплый, нежный и спокойный голос |
                | **Uncle_Fu** | Мужской | Китайский | Зрелый, солидный, глубокий мужской голос (рассказчик, аудиокниги) |
                | **Dylan** | Мужской | Китайский (Пекин) | Молодой, четкий, естественный мужской голос |
                | **Eric** | Мужской | Китайский (Сычуань) | Энергичный, живой мужской голос с легким хриплым оттенком |
                | **Ryan** | Мужской | Английский | Динамичный, ритмичный, современный мужской голос |
                | **Aiden** | Мужской | Английский | Солнечный американский мужской тембр, ясный средний регистр |
                | **Ono_Anna** | Женский | Японский | Игривый, звонкий, эмоциональный японский женский голос |
                | **Sohee** | Женский | Корейский | Теплый, богатый эмоциями корейский женский голос |

                ---
                ### Доступные модели и форматы (GGUF vs PyTorch)
                - **GGUF (Сверхбыстрый нативный C++ движок с аппаратным ускорением CUDA RTX 5080):**
                  - **Q8_0 (~2.1 GB для 1.7B / ~940 MB для 0.6B):** Рекомендуется! Качество неотличимо от оригинала при 2-3x меньшем расходе памяти и генерации за доли секунды.
                  - **Q4_K_M (~1.2 GB для 1.7B / ~600 MB для 0.6B):** Максимальная скорость и минимальный размер. Идеально для мгновенной генерации речи.
                  - **BF16 (~3.9 GB для 1.7B / ~1.8 GB для 0.6B):** 16-битная точность без квантования в GGUF формате.
                - **Оригинал (PyTorch / Hugging Face SafeTensors):**
                  - Стандартные веса 1.7B и 0.6B через библиотеку PyTorch.
                - **Поддержка эмоций и темпа (Instruct):**
                  - Модели 1.7B (как GGUF, так и PyTorch) полноценно поддерживают эмоции и пресеты интонаций.
                *Выбранная GGUF-модель автоматически скачивается из репозитория Serveurperso/Qwen3-TTS-GGUF в папку model/gguf/ при первом вызове.*

                ---
                ### Подсказки по использованию Instruct (Эмоции и стиль)
                - Инструкции можно указывать как на английском, так и на русском языке (например, **"Very happy and energetic"**, **"Шёпотом, таинственно"**, **"Строгий официальный тон"**).
                - Модель поддерживает комбинированные описания тембра, скорости и эмоций.
                - При значении языка **Auto** модель автоматически определяет язык по введённому тексту.

                ---
                ### Структура каталогов проекта
                - Веса GGUF: `model/gguf/`
                - Веса PyTorch: `model/`
                - Бинарный движок C++: `bin/qwen-tts.exe`
                - Сохранённые аудио: `outputs/`
                """
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-TTS WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Хост (по умолчанию 127.0.0.1, используйте 0.0.0.0 для локальной сети)")
    parser.add_argument("--port", type=int, default=7860, help="Порт (по умолчанию 7860)")
    parser.add_argument("--share", action="store_true", default=True, help="Создать внешнюю публичную ссылку Gradio (по умолчанию ВКЛЮЧЕНО)")
    parser.add_argument("--no-share", action="store_false", dest="share", help="Отключить внешнюю ссылку (только локально)")
    parser.add_argument("--preload", action="store_true", default=False, help="Предзагрузка модели PyTorch при старте")
    parser.add_argument("--model-size", type=str, default="1.7B", choices=["1.7B", "0.6B"], help="Размер модели для предзагрузки PyTorch (по умолчанию 1.7B)")
    args = parser.parse_args()

    print("=========================================================")
    print("               Запуск Qwen3-TTS WebUI                    ")
    print(f"  Устройство: {DEVICE} ({DTYPE})")
    print(f"  Папка моделей: {MODELS_DIR}")
    print(f"  Размер модели по умолчанию: {args.model_size}")
    print(f"  Локальный адрес: http://127.0.0.1:{args.port}")
    if args.share:
        print("  Внешний доступ: ВКЛЮЧЕН (создается публичная ссылка gradio.live)")
    print("=========================================================")

    if args.preload:
        try:
            print(f"[+] Предзагрузка модели CustomVoice ({args.model_size})...")
            load_engine("CustomVoice", args.model_size)
            print("[+] Предзагрузка успешно завершена!")
        except Exception as e:
            print(f"[-] Предупреждение при предзагрузке: {e}")

    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=True,
        )
    except OSError as e:
        if "Cannot find empty port" in str(e):
            print(f"[!] Порт {args.port} уже занят. Поиск следующего свободного порта...")
            demo.launch(
                server_name=args.host,
                server_port=None,
                share=args.share,
                inbrowser=True,
            )
        else:
            raise e
