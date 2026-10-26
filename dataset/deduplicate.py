"""
deduplicate.py - Сквозная дедупликация и формирование честного Train/Val/Test сплита
для датасета «РВК Safety» (Росводоканал).
Использует высокоскоростной dHash (Difference Hash) без тяжелых зависимостей OpenBLAS.
"""

import json
import time
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

def calculate_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """
    Вычисляет 64-битный Difference Hash (dHash) через Pillow.
    Не зависит от внешних библиотек OpenBLAS / SciPy, потребляет минимум RAM.
    """
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(small.getdata())
    diff_bits = 0
    bit_index = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            if left > right:
                diff_bits |= (1 << bit_index)
            bit_index += 1
    return diff_bits

def hamming_dist(h1: int, h2: int) -> int:
    """Расстояние Хэмминга между двумя 64-битными хешами (число несовпадающих бит)."""
    return bin(h1 ^ h2).count("1")

def extract_date(filename: str) -> str:
    """Извлекает дату съемки из имени файла (например, 2026__08__10)."""
    parts = filename.split("__")
    if len(parts) >= 3:
        return f"{parts[0]}__{parts[1]}__{parts[2]}"
    return "unknown"

def process_dataset(
    raw_dirs: list[Path],
    output_manifest: Path,
    hash_threshold: int = 4,
    val_dates: set[str] = None,
    test_dates: set[str] = None
):
    start_time = time.time()
    print("=== Начало сквозной дедупликации датасета РВК (dHash) ===")
    
    if val_dates is None:
        val_dates = {"2026__08__29", "2026__08__30", "2026__08__31"}
    if test_dates is None:
        test_dates = {"2026__09__01", "2026__09__02", "2026__09__03", "2026__09__04"}

    all_files = []
    for d in raw_dirs:
        if d.exists():
            files = sorted(list(d.glob("*.jpg")))
            all_files.extend(files)
            print(f"Найдено в {d.name}: {len(files)} файлов")
        else:
            print(f"Предупреждение: папка {d} не найдена!")

    total_files = len(all_files)
    print(f"Всего файлов для обработки: {total_files}")

    by_date = defaultdict(list)
    for f in all_files:
        d = extract_date(f.name)
        by_date[d].append(f)

    print(f"Уникальных съемочных дней: {len(by_date)}")

    unique_files = []
    duplicates_map = {}
    hashes = {}  # hash (int) -> canonical_path (Path)

    print("\nВычисление dHash и поиск дубликатов...")
    for date_key, files in sorted(by_date.items()):
        for img_path in tqdm(files, desc=f"День {date_key}"):
            try:
                with Image.open(img_path) as img:
                    h = calculate_dhash(img)
                    
                    is_dup = False
                    for existing_h, canonical_path in hashes.items():
                        if hamming_dist(h, existing_h) <= hash_threshold:
                            duplicates_map[str(img_path)] = str(canonical_path)
                            is_dup = True
                            break
                    
                    if not is_dup:
                        hashes[h] = img_path
                        unique_files.append(img_path)
            except Exception as e:
                print(f"Ошибка чтения {img_path.name}: {e}")

    splits = {"train": [], "val": [], "test": []}
    for f in unique_files:
        d = extract_date(f.name)
        path_str = str(f)
        if d in test_dates:
            splits["test"].append(path_str)
        elif d in val_dates:
            splits["val"].append(path_str)
        else:
            splits["train"].append(path_str)

    elapsed = round(time.time() - start_time, 2)
    print("\n=== Результаты дедупликации ===")
    print(f"Обработано всего:    {total_files}")
    print(f"Уникальных кадров:   {len(unique_files)} ({round(len(unique_files)/total_files*100, 1)}%)")
    print(f"Отсеяно дубликатов:  {len(duplicates_map)} ({round(len(duplicates_map)/total_files*100, 1)}%)")
    print("Сплиты (без утечки данных по датам):")
    print(f"  - Train: {len(splits['train'])} кадров")
    print(f"  - Val:   {len(splits['val'])} кадров")
    print(f"  - Test:  {len(splits['test'])} кадров")
    print(f"Время обработки: {elapsed} сек")

    manifest = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "dHash (difference hash 64-bit)",
            "total_scanned": total_files,
            "unique_count": len(unique_files),
            "duplicates_count": len(duplicates_map),
            "hash_threshold": hash_threshold,
            "val_dates": list(val_dates),
            "test_dates": list(test_dates)
        },
        "splits": splits,
        "duplicates": duplicates_map
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nМанифест сохранен: {output_manifest}")

if __name__ == "__main__":
    base = Path(".")
    raw_dirs = [
        base / "rvk-ai-vision-all-raw-20260904T132729Z-part-001" / "raw",
        base / "rvk-ai-vision-all-raw-20260904T132729Z-part-002" / "raw",
        base / "rvk-ai-vision-all-raw-20260904T132729Z-part-003" / "raw",
    ]
    manifest_path = base / "dataset" / "splits_manifest.json"
    process_dataset(raw_dirs, manifest_path, hash_threshold=4)
