# Qwen3-TTS WebUI

[Русский](#русский) | [English](#english)

---

## English

A high-performance local WebUI for speech synthesis and voice cloning based on the **Qwen3-TTS** model family. The application supports two inference backends: a native C++ GGUF engine with hardware CUDA acceleration and a standard PyTorch implementation.

### Key Capabilities

- **Multilingual Synthesis:** Native support for Russian, English, Chinese, Japanese, Korean, German, French, Spanish, Portuguese, Italian, and automatic language detection.
- **Dual Inference Backends:**
  - **Native C++ (GGUF):** Sub-second generation via CUDA with reduced VRAM consumption (quantizations: Q8_0, Q4_K_M, and BF16).
  - **PyTorch (SafeTensors):** Standard Hugging Face transformer backend supporting both 1.7B and 0.6B model sizes.
- **Instruction-Guided Prosody (Instruct Tuning):** Dynamic tone and emotion modulation for 1.7B models using natural language instructions. Includes 17 built-in presets (Calm, Joyful, Sad, Angry, Whisper, Seductive, Anchor, Storyteller, Confident, Enthusiastic, Ironic, Mysterious, Fast, Slow, Tender, Tired, Anxious).
- **Speaker Filtering by Gender:** Quick filtering of 9 pre-trained speakers by gender (Female, Male, All) with detailed acoustic profile descriptions.
- **Zero-Shot Voice Cloning:** Clone target voices from 3+ seconds of reference audio. Supports both Full In-Context Learning (with reference transcript) and acoustic timbre mode (X-Vector extraction without transcript).
- **Automatic Audio Token Allocation:** Automatic calculation of the optimal token budget based on input text length.

---

### System Requirements

- **Operating System:** Windows 10/11 (64-bit) or Linux (x86_64).
- **GPU:** NVIDIA GPU with CUDA support (Compute Capability 7.0+, recommended: RTX 3000/4000/5000 series with 8+ GB VRAM).
- **RAM:** Minimum 8 GB (16 GB recommended).
- **Python:** Version 3.10 to 3.13.

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Daler-Karimov/Qwen-TTS.git
   cd Qwen-TTS
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Quick Start

- **Windows:** Run `start.bat` or execute:
  ```bash
  python webui.py
  ```
- **CLI Options:**
  ```bash
  python webui.py --host 127.0.0.1 --port 7860 --no-share
  ```
  - `--host`: Bind host (default: `127.0.0.1`, use `0.0.0.0` for local network access).
  - `--port`: Port number (default: `7860`).
  - `--share`: Generate a public Gradio link.
  - `--no-share`: Run locally only (default in scripts).
  - `--preload`: Preload the default PyTorch model during startup.
  - `--model-size`: Default model size (`1.7B` or `0.6B`).

Upon first launch with a GGUF model variant, the required weights are automatically downloaded from Hugging Face (`Serveurperso/Qwen3-TTS-GGUF`) into the local `model/gguf/` directory.

---

### Pre-Trained Speakers Reference

| Speaker | Gender | Native Language | Acoustic Characteristics |
|---|---|---|---|
| Vivian | Female | Chinese | Bright, clear, young female voice with crisp articulation |
| Serena | Female | Chinese | Warm, gentle, soft, and friendly tone |
| Uncle_Fu | Male | Chinese | Mature, deep, resonant narrator voice (ideal for audiobooks) |
| Dylan | Male | Chinese (Beijing) | Young, clear, natural male delivery with neutral diction |
| Eric | Male | Chinese (Sichuan) | Lively, energetic male voice with a slight rasp |
| Ryan | Male | English | Dynamic, rhythmic modern male voice with strong cadence |
| Aiden | Male | English | Warm American male voice with balanced mid register |
| Ono_Anna | Female | Japanese | Playful, melodic Japanese female timbre |
| Sohee | Female | Korean | Expressive, emotional Korean female voice |

---

### Project Structure

```text
Qwen-TTS/
├── bin/                    # Precompiled GGUF C++ binaries and CUDA runtime libraries
├── model/                  # Local model cache (PyTorch and GGUF weights, git-ignored)
├── outputs/                # Generated audio outputs (git-ignored)
├── requirements.txt        # Python package dependencies
├── run.py                  # Standalone CLI inference script
├── start.bat               # Windows one-click startup batch script
├── webui.py                # Main Gradio application with dual backends
└── README.md               # Documentation (English and Russian)
```

---

## Русский

Высокопроизводительный локальный веб-интерфейс для синтеза речи и клонирования голоса на базе семейства моделей **Qwen3-TTS**. Приложение поддерживает два режима исполнения: нативный C++ движок GGUF с аппаратным ускорением CUDA и стандартную реализацию на PyTorch.

### Основные возможности

- **Мультиязычный синтез:** Поддержка русского, английского, китайского, японского, корейского, немецкого, французского, испанского, португальского, итальянского языков, а также автоматическое определение языка.
- **Два движка генерации:**
  - **Native C++ (GGUF):** Генерация за доли секунды через CUDA с минимальным потреблением видеопамяти (квантования: Q8_0, Q4_K_M и BF16).
  - **PyTorch (SafeTensors):** Стандартный движок на базе Hugging Face transformers с поддержкой моделей 1.7B и 0.6B.
- **Управление интонацией и эмоциями (Instruct Tuning):** Прямое управление тоном речи моделей 1.7B через текстовые инструкции. Включает 17 встроенных пресетов (Спокойно, Радостно, Грустно, Сердито, Шёпот, Соблазнительно, Диктор, Рассказчик, Уверенно, Восторженно, Иронично, Загадочно, Быстро, Медленно, Нежно, Устало, Тревожно).
- **Фильтрация дикторов по полу:** Быстрый отбор 9 предустановленных дикторов по полу (Женский, Мужской, Все) с отображением описания тембра.
- **Клонирование голоса (Zero-Shot Voice Clone):** Клонирование голоса по аудио-образцу от 3 секунд. Поддерживаются режимы с транскриптом (Full ICL) и без транскрипта (X-Vector режим тембра).
- **Автоматический расчёт токенов:** Автоматическое определение требуемой длины аудио-токенов в зависимости от объема входного текста.

---

### Системные требования

- **Операционная система:** Windows 10/11 (64-бит) или Linux (x86_64).
- **Видеокарта:** NVIDIA с поддержкой CUDA (Compute Capability 7.0+, рекомендуется RTX 3000/4000/5000 с объемом видеопамяти от 8 ГБ).
- **Оперативная память:** от 8 ГБ (рекомендуется 16 ГБ).
- **Python:** версии 3.10 - 3.13.

---

### Установка

1. **Клонирование репозитория:**
   ```bash
   git clone https://github.com/Daler-Karimov/Qwen-TTS.git
   cd Qwen-TTS
   ```

2. **Создание и активация виртуального окружения:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux:
   source venv/bin/activate
   ```

3. **Установка зависимостей:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Запуск

- **Windows:** Запустите `start.bat` или выполните команду:
  ```bash
  python webui.py
  ```
- **Параметры запуска:**
  ```bash
  python webui.py --host 127.0.0.1 --port 7860 --no-share
  ```
  - `--host`: Сетевой адрес (по умолчанию: `127.0.0.1`, используйте `0.0.0.0` для доступа по локальной сети).
  - `--port`: Порт (по умолчанию: `7860`).
  - `--share`: Создать публичную ссылку Gradio.
  - `--no-share`: Запуск только локально.
  - `--preload`: Предзагрузка модели PyTorch при старте.
  - `--model-size`: Размер модели по умолчанию (`1.7B` или `0.6B`).

При первом выборе варианта модели GGUF необходимые файлы автоматически загружаются с Hugging Face (`Serveurperso/Qwen3-TTS-GGUF`) в локальную папку `model/gguf/`.

---

### Таблица доступных дикторов

| Диктор | Пол | Родной язык | Характеристика тембра |
|---|---|---|---|
| Vivian | Женский | Китайский | Яркий, чистый, молодой женский голос с четкой дикцией |
| Serena | Женский | Китайский | Мягкий, теплый, нежный и спокойный голос |
| Uncle_Fu | Мужской | Китайский | Зрелый, глубокий мужской голос рассказчика (аудиокниги) |
| Dylan | Мужской | Китайский (Пекин) | Молодой, четкий, естественный мужской голос |
| Eric | Мужской | Китайский (Сычуань) | Энергичный, живой мужской голос с легкой хрипотцой |
| Ryan | Мужской | Английский | Динамичный, ритмичный, современный мужской голос |
| Aiden | Мужской | Английский | Солнечный американский мужской тембр, ясный средний регистр |
| Ono_Anna | Женский | Японский | Игривый, звонкий японский женский голос |
| Sohee | Женский | Корейский | Теплый, богатый эмоциями корейский женский голос |

---

### Лицензия и зависимости

Модели и веса распространяются в соответствии с условиями лицензии Qwen. Компиляция C++ движка GGUF основана на проекте `qwentts.cpp` (GGML).
