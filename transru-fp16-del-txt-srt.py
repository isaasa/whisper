import os
import whisper
import torch

def format_timestamp(seconds):
    """
    Форматирует время в секундах в строку формата hh:mm:ss,ms для SRT.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def save_srt_file(result, srt_output_file):
    """
    Сохраняет транскрипцию в формате .srt, используя сегменты результата транскрипции.
    
    :param result: Словарь результата транскрипции, содержащий ключ 'segments'.
    :param srt_output_file: Путь для сохранения .srt файла.
    """
    segments = result.get('segments', [])
    if not segments:
        print("Нет сегментов для сохранения в формате SRT.")
        return

    srt_lines = []
    for idx, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.get('start', 0))
        end_time = format_timestamp(segment.get('end', 0))
        text = segment.get('text', '').strip()
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(f"{text}\n")

    # Убедимся, что выходная директория существует
    os.makedirs(os.path.dirname(srt_output_file), exist_ok=True)
    with open(srt_output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_lines))
    print(f"SRT транскрипция сохранена в {srt_output_file}\n")

def transcribe_audio_to_text(model, input_file, output_file, language='ru'):
    """
    Транскрибирует аудио файл и сохраняет текстовую транскрипцию в формате .txt 
    и субтитры в формате .srt.

    :param model: Загруженная модель Whisper.
    :param input_file: Путь к входному .mp3 файлу.
    :param output_file: Путь для сохранения транскрипции в .txt файле.
    :param language: Язык транскрипции (по умолчанию 'ru' для русского).
    """
    try:
        result = model.transcribe(
            input_file,
            language=language,
            verbose=True,
            fp16=True,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4]
        )
    except Exception as e:
        print(f"Ошибка при транскрипции файла {input_file}: {e}")
        raise Exception(f"Транскрипция для {input_file} не удалась.")

    # Извлекаем текст транскрипции и заменяем переносы строк на пробелы
    transcription = result.get('text', '').replace('\n', ' ').strip()
    print(f"Файл: {os.path.basename(input_file)} | Транскрипция: {transcription}")

    # Убедимся, что выходная директория существует
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Сохраняем транскрипцию в текстовый файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription + '\n')
    print(f"Транскрипция сохранена в {output_file}\n")

    # Формируем путь для .srt файла (с той же базой имени файла)
    srt_output_file = os.path.splitext(output_file)[0] + ".srt"
    save_srt_file(result, srt_output_file)

    # После успешного сохранения транскрипции удаляем аудиофайл
    try:
        os.remove(input_file)
        print(f"Аудиофайл {os.path.basename(input_file)} удалён.\n")
    except Exception as e:
        print(f"Не удалось удалить аудиофайл {input_file}: {e}\n")

def batch_transcribe(input_folder='audio', output_folder='result', language='ru'):
    """
    Пакетная транскрипция всех .mp3 файлов в указанной папке и её подпапках.
    Сохраняет транскрипции в формате .txt и .srt.

    :param input_folder: Путь к входной папке с .mp3 файлами.
    :param output_folder: Путь к выходной папке для сохранения транскрипций.
    :param language: Язык транскрипции (по умолчанию 'ru' для русского).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("CUDA обнаружен. Используется GPU для транскрипции.")
    else:
        print("CUDA не обнаружен. Используется CPU для транскрипции.")

    print("Загрузка модели Whisper...")
    model = whisper.load_model("medium", device=device)
    print("Модель загружена.\n")

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.mp3'):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                base_name = os.path.splitext(file)[0]
                output_file = os.path.join(output_dir, f"{base_name}.txt")
                try:
                    transcribe_audio_to_text(model, input_file, output_file, language=language)
                except Exception as e:
                    print(f"Не удалось транскрибировать {input_file}: {e}\n")
                    continue

    print("Пакетная транскрипция завершена.")

if __name__ == "__main__":
    input_folder = 'audio'        # Папка с подпапками и .mp3 файлами
    output_folder = 'result'      # Папка для сохранения .txt и .srt транскрипций
    transcription_language = 'ru' # Язык транскрипции

    batch_transcribe(input_folder=input_folder, output_folder=output_folder, language=transcription_language)
