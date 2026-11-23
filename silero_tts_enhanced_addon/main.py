import os
import re
import wave
import tempfile
import shutil
import array
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

# === НАСТРОЙКА ПОСТОЯННОЙ ПАМЯТИ (ЧТОБЫ МОДЕЛИ НЕ КАЧАЛИСЬ ПРИ РЕСТАРТЕ) ===
PERSISTENT_DIR = "/data/silero_cache"

if not os.path.exists("/data"):
    PERSISTENT_DIR = "./silero_cache"
    
os.makedirs(PERSISTENT_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = PERSISTENT_DIR

# Безопасный поиск папки и создание симлинка
try:
    # Импортируем конкретный файл, у которого точно есть путь
    from silero_tts import silero_tts as st_module
    lib_dir = os.path.dirname(st_module.__file__)
    models_dir = os.path.join(lib_dir, "silero_models")

    # Если это еще не симлинк, удаляем папку и создаем ссылку на /data
    if not os.path.islink(models_dir):
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
        os.symlink(PERSISTENT_DIR, models_dir)
        print(f"Симлинк кэша успешно создан: {models_dir} -> {PERSISTENT_DIR}")
except Exception as e:
    print(f"Внимание: Не удалось настроить жесткий кэш моделей: {e}")
# ===========================================================================

from silero_tts.silero_tts import SileroTTS

# === ПРЕДОБРАБОТКА ТЕКСТА ДЛЯ TTS ===
# Silero ломает интонацию на парных запятых вокруг вводных слов, на скобках,
# на markdown и на мусорных сочетаниях знаков препинания. Здесь всё это
# приводится к виду, который синтезатор читает ровно.

# Как читать служебный маркер ".?" в конце ответа (он нужен HA, а не голосу):
#   False -> ".?" превращается в "." (ровная интонация)
#   True  -> ".?" превращается в "?" (вопросительная интонация)
MARKER_AS_QUESTION = False

# Список вводных слов можно дополнять: сравнение идёт без учёта регистра,
# слова берутся только в конструкции "запятая + слово + запятая".
INTRO_WORDS = [
    # обороты из нескольких слов
    "к сожалению", "к счастью", "кроме того", "более того", "таким образом",
    "как правило", "в свою очередь", "честно говоря", "короче говоря",
    "к тому же", "в общем", "в целом", "в принципе", "в частности",
    "в конце концов", "по крайней мере", "по правде говоря", "если честно",
    "мягко говоря", "прямо скажем", "между нами", "как видишь", "как известно",
    "как ни странно", "как ни жаль", "само собой", "надо думать", "ясное дело",
    "по сути", "по-моему", "по-твоему", "по-видимому",
    "во-первых", "во-вторых", "в-третьих",
    # одиночные слова
    "наверное", "возможно", "вероятно", "видимо", "очевидно", "похоже",
    "действительно", "собственно", "например", "кстати", "конечно", "итак",
    "впрочем", "однако", "правда", "причём", "причем", "словом", "кажется",
    "пожалуй", "разумеется", "безусловно", "естественно", "напротив",
    "наоборот", "точнее", "вернее", "буквально", "увы", "признаться",
    # глаголы-вставки от первого лица и обращения к собеседнику
    "надеюсь", "признаюсь", "полагаю", "думаю", "считаю", "замечу",
    "скажем", "допустим", "знаешь", "понимаешь", "поверь", "представь",
    "согласись", "заметь",
]

_INTRO_RE = re.compile(
    r",\s*(" + "|".join(sorted(map(re.escape, INTRO_WORDS), key=len, reverse=True)) + r")\s*,",
    re.IGNORECASE,
)

# Эмодзи и пиктограммы: синтезатор их либо пропускает с паузой, либо читает вслух
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # эмодзи, пиктограммы
    "\U00002600-\U000027BF"   # символы и дингбаты
    "\U0001F1E6-\U0001F1FF"   # флаги
    "\u2190-\u21FF"           # стрелки
    "\u2B00-\u2BFF"           # разные символы
    "\uFE0F\u200D\u20E3"      # модификаторы эмодзи
    "]+"
)

def prettify_for_tts(text: str) -> str:
    """Готовит текст к синтезу: убирает конструкции, на которых Silero ломает интонацию."""

    # 1. Парные запятые вокруг вводного слова -> тире ("облачно — кстати — температура")
    text = _INTRO_RE.sub(r" — \1 — ", text)

    # 2. Скобки -> тире, содержимое сохраняем
    text = re.sub(r"\s*\(([^)]*)\)", r" — \1 — ", text)

    # 3. Markdown: ссылки, URL, жирный, заголовки, списки, маркеры
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "ссылка", text)
    text = re.sub(r"[*_#`>|]+", " ", text)
    text = re.sub(r"[•·▪◦]+", " ", text)

    # 4. Многоточие -> точка
    text = text.replace("…", ".").replace("...", ".")

    # 5. Эмодзи
    text = _EMOJI_RE.sub("", text)

    # 6. Знаки препинания: схлопываем серии и мусорные сочетания.
    #    Служебный маркер ".?" (нужен HA для продолжения диалога) для голоса
    #    превращаем либо в "?", либо в "." — см. MARKER_AS_QUESTION.
    if MARKER_AS_QUESTION:
        text = re.sub(r"\.\s*\?+", "?", text)
    text = re.sub(r"([!?.])\1+", r"\1", text)          # "!!!" -> "!", ".." -> "."
    text = re.sub(r"([.!?])\s*[.!?]+", r"\1", text)    # ".?" и "!?" -> первый знак
    text = re.sub(r"([.!?])\s*[,;:]+", r"\1", text)    # ".," -> "."
    text = re.sub(r"([,;:])\s*[.!?]+", r"\1", text)    # ",." -> ","
    text = re.sub(r"[,;:]{2,}", ",", text)             # ",," -> ","
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)       # пробел перед знаком

    # 7. Тире: нормализуем пробелы, убираем двойные и висячие перед знаком
    text = re.sub(r"\s*[—–]\s*", " — ", text)
    text = re.sub(r"(\s—\s*){2,}", " — ", text)
    text = re.sub(r"\s*—\s*(?=[.!?,;:])", "", text)
    text = re.sub(r"^[\s—]+", "", text)
    text = re.sub(r"[\s—]+$", "", text)

    # 8. Лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === СЛОВАРЬ УДАРЕНИЙ-ИСКЛЮЧЕНИЙ ===
# Формат: "слово в нижнем регистре": "слово с + перед ударной гласной",
# Каждая строка заканчивается запятой. Ударения сверяйте по орфоэпическому словарю.
STRESS_DICT = {
    "второй": "втор+ой",
    "второго": "втор+ого",
    "второму": "втор+ому",
    "вторым": "втор+ым",
    "втором": "втор+ом",
    "вторая": "втор+ая",
    "вторую": "втор+ую",
    "второе": "втор+ое",
    "вторые": "втор+ые",
    "вторых": "втор+ых",
    "вторыми": "втор+ыми",
}

_STRESS_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(STRESS_DICT, key=len, reverse=True))) + r")\b",
    re.IGNORECASE,
)

def apply_stress_dict(text: str) -> str:
    def repl(m):
        word = m.group(0)
        fixed = STRESS_DICT[word.lower()]
        return fixed[0].upper() + fixed[1:] if word[0].isupper() else fixed
    return _STRESS_RE.sub(repl, text)

def split_clauses(text: str):
    return [p for p in re.split(r"(?<=[.!?…,;:])\s+", text) if p.strip()]

def join_wavs(paths, out_path, pause_ms=100, fade_ms=15):
    with wave.open(out_path, "wb") as out:
        params = None
        for p in paths:
            with wave.open(p, "rb") as w:
                if params is None:
                    params = w.getparams()
                    out.setparams(params)
                rate = w.getframerate()
                ch = w.getnchannels()
                width = w.getsampwidth()
                raw = w.readframes(w.getnframes())

            if width != 2:
                # неожиданный формат: склеиваем как раньше
                out.writeframes(raw)
                continue

            data = array.array("h")
            data.frombytes(raw)
            n = min(rate * fade_ms // 1000, len(data) // ch // 2)
            for i in range(n):
                k = i / n
                for c in range(ch):
                    data[i * ch + c] = int(data[i * ch + c] * k)
                    j = len(data) - 1 - (i * ch + (ch - 1 - c))
                    data[j] = int(data[j] * k)
            out.writeframes(data.tobytes())
            out.writeframes(b"\x00" * (rate * pause_ms // 1000 * ch * 2))

# ===================================

app = FastAPI()

# === ПРИНУДИТЕЛЬНАЯ МОДЕЛЬ ===
# None      -> берём то, что прислала HA-интеграция (сейчас v5_ru)
# "v5_5_ru" -> принудительно, настройки интеграции игнорируются
FORCE_MODEL_ID = "v5_5_ru"

# Пауза между кусками при склейке WAV, миллисекунды.
# Используется только когда текст режется на фразы ради ручных ударений.
PAUSE_BETWEEN_CLAUSES_MS = 100

tts_engine = None
current_model = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "aidar"
    language: str = "ru"
    model_id: str = "v5_ru"
    sample_rate: int = 48000
    put_accent: bool = True
    put_yo: bool = True

@app.post("/tts")
def generate_tts(req: TTSRequest):
    global tts_engine, current_model
    try:
        # Принудительная модель перебивает то, что прислала HA-интеграция.
        if FORCE_MODEL_ID:
            req.model_id = FORCE_MODEL_ID

        if tts_engine is None or current_model != req.model_id:
            print(f"Загрузка модели: {req.model_id}")
            try:
                tts_engine = SileroTTS(model_id=req.model_id, language=req.language, speaker=req.voice)
                current_model = req.model_id
            except Exception as model_err:
                print(f"Не удалось загрузить {req.model_id}: {model_err}")
                if req.model_id != "v5_ru":
                    print("Откат на v5_ru")
                    req.model_id = "v5_ru"
                    tts_engine = SileroTTS(model_id="v5_ru", language=req.language, speaker=req.voice)
                    current_model = "v5_ru"
                else:
                    raise

        tts_engine.put_accent = req.put_accent
        tts_engine.put_yo = req.put_yo

        if getattr(tts_engine, 'language', '') != req.language:
            tts_engine.change_language(req.language)
        if getattr(tts_engine, 'speaker', '') != req.voice:
            tts_engine.change_speaker(req.voice)
            
        if hasattr(tts_engine, 'change_sample_rate'):
            tts_engine.change_sample_rate(req.sample_rate)

        # === ПРЕДОБРАБОТКА: обязательно ДО словаря ударений ===
        clean_text = prettify_for_tts(req.text)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        print(f"Исходный текст: '{req.text}'")
        print(f"Текст на синтез: '{clean_text}'")
        print(f"Модель: {current_model}, голос={req.voice}")

        if not _STRESS_RE.search(clean_text):
            # Словарных слов нет: поведение как раньше
            tts_engine.tts(clean_text, temp_path)
        else:
            part_paths = []
            try:
                for clause in split_clauses(clean_text):
                    fixed = apply_stress_dict(clause)
                    manual = fixed != clause
                    # во фразах со словарным словом отключаем авторасстановку,
                    # чтобы ручной "+" не затирался
                    tts_engine.put_accent = False if manual else req.put_accent
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as pt:
                        part_paths.append(pt.name)
                    print(f"  фраза (ручное ударение={manual}): '{fixed}'")
                    tts_engine.tts(fixed, part_paths[-1])
                join_wavs(part_paths, temp_path, pause_ms=PAUSE_BETWEEN_CLAUSES_MS)
            finally:
                tts_engine.put_accent = req.put_accent
                for p in part_paths:
                    if os.path.exists(p):
                        os.remove(p)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception("Файл не сгенерировался (0 байт). Ошибка библиотеки Silero.")
            
        with open(temp_path, "rb") as f:
            audio = f.read()
            
        os.remove(temp_path)
        print(f"Успех! Отправлено {len(audio)} байт.")
        return Response(content=audio, media_type="audio/wav")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
