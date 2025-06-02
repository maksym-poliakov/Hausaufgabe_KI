
import os
import torch
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import argparse

def setup_whisper():
    """
    Устанавливает и настраивает Whisper, если это необходимо.
    Возвращает доступное устройство (CUDA или CPU).
    """
    try:
        import whisper
        print(f"Whisper version: {whisper.__version__}")
    except ImportError:
        print("Installing OpenAI Whisper...")
        os.system("pip install -U openai-whisper")
        try:
            import whisper
            print(f"Whisper version: {whisper.__version__}")
        except ImportError:
            print("Failed to install whisper. Please install manually:")
            print("pip install -U openai-whisper")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def load_model(model_size="tiny", device=None):
    """
    Загружает модель Whisper.

    Аргументы:
        model_size (str): Размер модели. Варианты: "tiny", "base", "small", "medium", "large"
        device (str): Устройство для использования (cuda или cpu)

    Возвращает:
        Загруженную модель
    """
    print(f"Loading Whisper {model_size} model...")
    try:
        import whisper
        if not hasattr(whisper, 'load_model'):
            print("Error: 'whisper' module does not have 'load_model'. Ensure you are using 'openai-whisper'.")
            print("Try reinstalling: pip install -U openai-whisper")
            sys.exit(1)
        model = whisper.load_model(model_size, device=device)
        print(f"Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def transcribe_audio_file(model, audio_path, language=None):
    """
    Транскрибирует аудиофайл с помощью Whisper.

    Аргументы:
        model: Модель Whisper
        audio_path (str): Путь к аудиофайлу
        language (str, опционально): Код языка (например, "en", "fr", "ja", "ru")

    Возвращает:
        dict: Результат транскрипции
    """
    print(f"Transcribing: {audio_path}")
    options = {}
    if language:
        options["language"] = language
    result = model.transcribe(audio_path, **options)
    return result

def save_transcription(result, output_file=None):
    """
    Сохраняет результат транскрипции в файл.

    Аргументы:
        result (dict): Результат транскрипции от Whisper
        output_file (str, опционально): Путь для сохранения результата

    Возвращает:
        str: Путь к сохраненному файлу
    """
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcription_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Transcription saved to: {output_file}")
    return output_file

def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    Записывает аудио с микрофона.

    Аргументы:
        duration (int): Длительность записи в секундах
        sample_rate (int): Частота дискретизации
        channels (int): Количество каналов (1=моно, 2=стерео)

    Возвращает:
        str: Путь к записанному аудиофайлу
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("Installing required packages for audio recording...")
        os.system("pip install sounddevice soundfile")
        import sounddevice as sd
        import soundfile as sf

    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        temp_file = f.name
    sf.write(temp_file, recording, sample_rate)
    print(f"Audio recorded and saved to temporary file: {temp_file}")
    return temp_file

def main():
    """
    Основная функция программы.
    Обрабатывает аргументы командной строки и управляет процессом транскрипции.
    """
    parser = argparse.ArgumentParser(description="Whisper Speech Recognition Tool")
    parser.add_argument("--file", type=str, help="Path to audio file for transcription")
    parser.add_argument("--record", type=int, default=0, help="Record audio for specified seconds")
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--language", type=str, help="Language code (e.g., 'en', 'fr')")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()
    device = setup_whisper()
    model = load_model(args.model, device)

    audio_path = None
    if args.file:
        audio_path = args.file
    elif args.record > 0:
        audio_path = record_audio(duration=args.record)
    else:
        print("No audio input specified. Use --file or --record")
        print("Example: python Hausaufgabe_8.py --file audio.mp3")
        print("Example: python Hausaufgabe_8.py --record 10")
        return

    result = transcribe_audio_file(model, audio_path, args.language)
    print("\nTranscription:")
    print(result["text"])

    if args.output:
        save_transcription(result, args.output)

    if args.record > 0 and audio_path:
        os.unlink(audio_path)
        print(f"Temporary audio file deleted: {audio_path}")

if __name__ == "__main__":
    main()
