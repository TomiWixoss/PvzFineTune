"""
Tool lọc ảnh trùng lặp sử dụng perceptual hashing.
Giữ lại ảnh có sự khác biệt đáng kể, loại bỏ frames gần giống nhau.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm


def compute_hash(image_path: str, hash_size: int = 16) -> imagehash.ImageHash:
    """Tính perceptual hash của ảnh."""
    img = Image.open(image_path)
    return imagehash.phash(img, hash_size=hash_size)


def find_duplicates(
    input_dir: str,
    threshold: int = 10,
    hash_size: int = 16
) -> dict[str, list[str]]:
    """
    Tìm các ảnh trùng lặp/gần giống nhau.
    
    Args:
        input_dir: Thư mục chứa ảnh
        threshold: Ngưỡng khác biệt (0=giống hệt, cao hơn=khác biệt hơn)
                   Recommend: 5-10 cho PvZ frames
        hash_size: Kích thước hash (16 cho độ chính xác cao)
    
    Returns:
        Dict với key là ảnh đại diện, value là list ảnh trùng
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Lấy tất cả ảnh
    images = []
    for ext in image_extensions:
        images.extend(Path(input_dir).glob(f'*{ext}'))
        images.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    images = sorted(images)
    print(f"Tìm thấy {len(images)} ảnh")
    
    # Tính hash cho tất cả ảnh
    hashes = {}
    print("Đang tính hash...")
    for img_path in tqdm(images):
        try:
            hashes[str(img_path)] = compute_hash(str(img_path), hash_size)
        except Exception as e:
            print(f"Lỗi với {img_path}: {e}")
    
    # Tìm duplicates
    duplicates = defaultdict(list)
    processed = set()
    
    print("Đang tìm ảnh trùng...")
    hash_list = list(hashes.items())
    
    for i, (path1, hash1) in enumerate(tqdm(hash_list)):
        if path1 in processed:
            continue
            
        for path2, hash2 in hash_list[i+1:]:
            if path2 in processed:
                continue
                
            # So sánh hash
            diff = hash1 - hash2
            if diff <= threshold:
                duplicates[path1].append(path2)
                processed.add(path2)
    
    return dict(duplicates)


def deduplicate(
    input_dir: str,
    output_dir: str,
    threshold: int = 10,
    keep_every_n: int = 1
) -> tuple[int, int]:
    """
    Lọc ảnh trùng và copy ảnh unique sang thư mục mới.
    
    Args:
        input_dir: Thư mục ảnh gốc
        output_dir: Thư mục output
        threshold: Ngưỡng khác biệt (5-10 recommend)
        keep_every_n: Giữ 1 trong mỗi n ảnh trùng (1=chỉ giữ 1)
    
    Returns:
        (số ảnh giữ lại, số ảnh loại bỏ)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    duplicates = find_duplicates(input_dir, threshold)
    
    # Tìm tất cả ảnh unique (không bị đánh dấu là duplicate)
    all_duplicates = set()
    for dups in duplicates.values():
        all_duplicates.update(dups)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = []
    for ext in image_extensions:
        all_images.extend(Path(input_dir).glob(f'*{ext}'))
        all_images.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    unique_images = [str(img) for img in all_images if str(img) not in all_duplicates]
    
    # Copy ảnh unique
    print(f"\nCopy {len(unique_images)} ảnh unique...")
    for img_path in tqdm(unique_images):
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(output_dir, filename))
    
    removed = len(all_images) - len(unique_images)
    print(f"\n✓ Giữ lại: {len(unique_images)} ảnh")
    print(f"✓ Loại bỏ: {removed} ảnh trùng")
    
    return len(unique_images), removed


def deduplicate_sequential(
    input_dir: str,
    output_dir: str,
    min_diff: int = 8
) -> tuple[int, int]:
    """
    Lọc ảnh theo thứ tự (tối ưu cho video frames).
    So sánh mỗi frame với frame trước đó.
    
    Args:
        input_dir: Thư mục ảnh gốc
        output_dir: Thư mục output  
        min_diff: Độ khác biệt tối thiểu để giữ frame (5-15 recommend)
    
    Returns:
        (số ảnh giữ lại, số ảnh loại bỏ)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    for ext in image_extensions:
        images.extend(Path(input_dir).glob(f'*{ext}'))
        images.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    images = sorted(images)  # Sort theo tên để giữ thứ tự
    print(f"Tìm thấy {len(images)} ảnh")
    
    if not images:
        return 0, 0
    
    kept = []
    last_hash = None
    
    print("Đang lọc ảnh...")
    for img_path in tqdm(images):
        try:
            current_hash = compute_hash(str(img_path))
            
            if last_hash is None:
                # Luôn giữ frame đầu tiên
                kept.append(img_path)
                last_hash = current_hash
            else:
                diff = current_hash - last_hash
                if diff >= min_diff:
                    kept.append(img_path)
                    last_hash = current_hash
                    
        except Exception as e:
            print(f"Lỗi với {img_path}: {e}")
    
    # Copy ảnh đã lọc
    print(f"\nCopy {len(kept)} ảnh...")
    for img_path in tqdm(kept):
        filename = img_path.name
        shutil.copy2(str(img_path), os.path.join(output_dir, filename))
    
    removed = len(images) - len(kept)
    print(f"\n✓ Giữ lại: {len(kept)} ảnh")
    print(f"✓ Loại bỏ: {removed} ảnh trùng")
    
    return len(kept), removed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lọc ảnh trùng lặp")
    parser.add_argument("input_dir", help="Thư mục chứa ảnh gốc")
    parser.add_argument("output_dir", help="Thư mục output")
    parser.add_argument("--threshold", type=int, default=8, 
                        help="Ngưỡng khác biệt (default: 8)")
    parser.add_argument("--sequential", action="store_true",
                        help="Dùng mode sequential (tối ưu cho video frames)")
    
    args = parser.parse_args()
    
    if args.sequential:
        deduplicate_sequential(args.input_dir, args.output_dir, args.threshold)
    else:
        deduplicate(args.input_dir, args.output_dir, args.threshold)
