import os
import whisper
import torch

def transcribe_audio_to_text(model, input_file, output_file, language='ru'):
    """
    Транскрибирует аудио файл и сохраняет текстовую транскрипцию.

    :param model: Загруженная модель Whisper.
    :param input_file: Путь к входному .mp3 файлу.
    :param output_file: Путь для сохранения транскрипции в .txt файле.
    :param language: Язык транскрипции (по умолчанию 'ru' для русского).
    """
    try:
        # Транскрибируем аудио с дополнительными параметрами:
        # - fp16=True: использование 16-битной точности на GPU (если доступно)
        # - beam_size=5 и best_of=5: расширенный поиск для повышения качества
        # - temperature: пробуем несколько значений, чтобы улучшить стабильность при наличии шума
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

    # После успешного сохранения транскрипции удаляем аудиофайл
    try:
        os.remove(input_file)
        print(f"Аудиофайл {os.path.basename(input_file)} удалён.\n")
    except Exception as e:
        print(f"Не удалось удалить аудиофайл {input_file}: {e}\n")


def batch_transcribe(input_folder='audio', output_folder='result', language='ru'):
    """
    Пакетная транскрипция всех .mp3 файлов в указанной папке и ее подпапках.

    :param input_folder: Путь к входной папке с .mp3 файлами.
    :param output_folder: Путь к выходной папке для сохранения транскрипций.
    :param language: Язык транскрипции (по умолчанию 'ru' для русского).
    """
    # Проверяем доступность CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("CUDA обнаружен. Используется GPU для транскрипции.")
    else:
        print("CUDA не обнаружен. Используется CPU для транскрипции.")

    # Загружаем модель Whisper один раз
    print("Загрузка модели Whisper...")
    model = whisper.load_model("medium", device=device)
    print("Модель загружена.\n")

    # Обходим входную папку рекурсивно
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.mp3'):
                input_file = os.path.join(root, file)
                # Определяем относительный путь для сохранения структуры папок
                relative_path = os.path.relpath(root, input_folder)
                # Формируем путь к выходной директории
                output_dir = os.path.join(output_folder, relative_path)
                # Задаем имя выходного файла с расширением .txt
                base_name = os.path.splitext(file)[0]
                output_file = os.path.join(output_dir, f"{base_name}.txt")
                try:
                    transcribe_audio_to_text(model, input_file, output_file, language=language)
                except Exception as e:
                    print(f"Не удалось транскрибировать {input_file}: {e}\n")
                    continue  # Переходим к следующему файлу

    print("Пакетная транскрипция завершена.")

if __name__ == "__main__":
    # Определяем входные и выходные директории
    input_folder = 'audio'        # Папка с подпапками и .mp3 файлами
    output_folder = 'result'      # Папка для сохранения .txt транскрипций

    # Указываем язык транскрипции (русский)
    transcription_language = 'ru'

    # Запускаем пакетную транскрипцию
    batch_transcribe(input_folder=input_folder, output_folder=output_folder, language=transcription_language)
