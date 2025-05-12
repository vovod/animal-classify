import os
import random
import shutil
from pathlib import Path


def create_asset_folder():
    """
    Tạo folder assets và copy một ảnh đại diện cho mỗi loài động vật
    """
    # Tạo thư mục assets nếu chưa có
    asset_dir = Path("assets")
    asset_dir.mkdir(exist_ok=True)

    # Đường dẫn đến thư mục dataset gốc
    animals_dir = Path("animals/animals")

    # Đọc danh sách động vật từ name.txt
    with open('name.txt', 'r') as f:
        animal_names = [line.strip() for line in f.readlines()]

    print("Bắt đầu tạo assets folder...")

    for animal_name in animal_names:
        # Thư mục chứa ảnh của từng loài
        animal_folder = animals_dir / animal_name

        if animal_folder.exists() and animal_folder.is_dir():
            # Lấy danh sách tất cả ảnh trong thư mục
            image_files = list(animal_folder.glob("*.jpg")) + \
                list(animal_folder.glob("*.jpeg")) + \
                list(animal_folder.glob("*.png"))

            if image_files:
                # Chọn ngẫu nhiên 1 ảnh
                selected_image = random.choice(image_files)

                # Tên file mới (chỉ lấy phần mở rộng của file gốc)
                new_filename = f"{animal_name}{selected_image.suffix}"
                destination_path = asset_dir / new_filename

                # Copy ảnh sang thư mục assets
                shutil.copy2(selected_image, destination_path)
                print(f"✓ Copied: {animal_name} -> {new_filename}")
            else:
                print(f"✗ No images found for: {animal_name}")
        else:
            print(f"✗ Folder not found: {animal_name}")

    print(
        f"\nHoàn thành! Đã tạo folder assets với {len(list(asset_dir.glob('*')))} ảnh.")
    print(f"Đường dẫn: {asset_dir.absolute()}")


def verify_assets():
    """
    Kiểm tra xem assets đã được tạo đầy đủ chưa
    """
    asset_dir = Path("assets")
    if not asset_dir.exists():
        print("Folder assets chưa được tạo!")
        return False

    # Đọc danh sách động vật từ name.txt
    with open('name.txt', 'r') as f:
        animal_names = [line.strip() for line in f.readlines()]

    missing_assets = []
    existing_assets = []

    for animal_name in animal_names:
        # Tìm ảnh của animal (có thể là .jpg, .jpeg hoặc .png)
        found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            asset_path = asset_dir / f"{animal_name}{ext}"
            if asset_path.exists():
                existing_assets.append(f"{animal_name}{ext}")
                found = True
                break

        if not found:
            missing_assets.append(animal_name)

    print(f"\nKiểm tra assets:")
    print(f"✓ Có sẵn: {len(existing_assets)}/90")
    print(f"✗ Thiếu: {len(missing_assets)}/90")

    if missing_assets:
        print("\nDanh sách ảnh thiếu:")
        for animal in missing_assets[:10]:  # Chỉ hiển thị 10 đầu
            print(f"  - {animal}")
        if len(missing_assets) > 10:
            print(f"  ... và {len(missing_assets) - 10} ảnh khác")

    return len(missing_assets) == 0


def create_missing_placeholders():
    """
    Tạo placeholder cho những ảnh bị thiếu
    """
    asset_dir = Path("assets")

    # Đọc danh sách động vật từ name.txt
    with open('name.txt', 'r') as f:
        animal_names = [line.strip() for line in f.readlines()]

    print("Tạo placeholder cho những ảnh bị thiếu...")

    for animal_name in animal_names:
        # Kiểm tra xem đã có ảnh chưa
        found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            if (asset_dir / f"{animal_name}{ext}").exists():
                found = True
                break

        if not found:
            # Tạo file placeholder
            placeholder_path = asset_dir / f"{animal_name}_placeholder.txt"
            with open(placeholder_path, 'w') as f:
                f.write(f"Placeholder for {animal_name}\n")
                f.write("Image not found in dataset\n")
                f.write("Please manually add an image for this animal\n")
            print(f"✓ Created placeholder: {animal_name}")


if __name__ == "__main__":
    print("TOOL TẠO ASSETS FOLDER")
    print("=" * 50)

    # Tạo assets folder
    create_asset_folder()

    # Kiểm tra kết quả
    print("\n" + "=" * 50)
    verify_assets()

    # Tạo placeholder cho những ảnh bị thiếu
    print("\n" + "=" * 50)
    create_missing_placeholders()

    print("\n" + "=" * 50)
    print("HOÀN THÀNH!")
    print("Giờ bạn có thể chạy GUI với assets folder mới.")
    print("Nếu có placeholder files, hãy thay thế chúng bằng ảnh thật.")
