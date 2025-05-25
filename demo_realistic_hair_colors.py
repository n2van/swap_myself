"""
Demo màu tóc thực tế cho Hair Segmentation
Sử dụng các màu tóc tự nhiên và phổ biến
"""

from hair_segmentation_simple import HairSegmentator
from hair_colors import *
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def demo_natural_hair_colors():
    """Demo với màu tóc tự nhiên"""
    print("=== DEMO MÀU TÓC TỰ NHIÊN ===")
    
    segmentator = HairSegmentator()
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists(model_path):
        print("❌ Cần có model để chạy demo")
        return
    
    segmentator.load_model(model_path)
    
    # Tìm ảnh test
    test_images = ["test_image.jpg", "test_image.png", "input.jpg"]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ Cần có ảnh test để chạy demo")
        return
    
    print(f"📸 Sử dụng ảnh: {image_path}")
    
    # Phân đoạn tóc một lần
    hair_mask = segmentator.segment_hair(image_path)
    
    # Lấy màu tóc tự nhiên
    natural_colors = get_natural_colors()
    
    print("🎨 Đang tạo ảnh với màu tóc tự nhiên...")
    
    for color_name, color_value in natural_colors.items():
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        output_path = f"natural_{color_name}.jpg"
        cv2.imwrite(output_path, colored_image)
        print(f"   ✅ Tạo {output_path} - RGB{color_value}")

def demo_trendy_hair_colors():
    """Demo với màu tóc xu hướng"""
    print("\n=== DEMO MÀU TÓC XU HƯỚNG ===")
    
    segmentator = HairSegmentator()
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists(model_path):
        print("❌ Cần có model để chạy demo")
        return
    
    segmentator.load_model(model_path)
    
    # Tìm ảnh test
    test_images = ["test_image.jpg", "test_image.png", "input.jpg"]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ Cần có ảnh test để chạy demo")
        return
    
    # Phân đoạn tóc một lần
    hair_mask = segmentator.segment_hair(image_path)
    
    # Lấy màu tóc xu hướng
    trendy_colors = get_trendy_colors()
    
    print("🌈 Đang tạo ảnh với màu tóc xu hướng...")
    
    for color_name, color_value in trendy_colors.items():
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        output_path = f"trendy_{color_name}.jpg"
        cv2.imwrite(output_path, colored_image)
        print(f"   ✅ Tạo {output_path} - RGB{color_value}")

def demo_popular_dye_colors():
    """Demo với màu tóc nhuộm phổ biến"""
    print("\n=== DEMO MÀU TÓC NHUỘM PHỔ BIẾN ===")
    
    segmentator = HairSegmentator()
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists(model_path):
        print("❌ Cần có model để chạy demo")
        return
    
    segmentator.load_model(model_path)
    
    # Tìm ảnh test
    test_images = ["test_image.jpg", "test_image.png", "input.jpg"]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ Cần có ảnh test để chạy demo")
        return
    
    # Phân đoạn tóc một lần
    hair_mask = segmentator.segment_hair(image_path)
    
    # Lấy màu tóc nhuộm
    dye_colors = get_dye_colors()
    
    print("💄 Đang tạo ảnh với màu tóc nhuộm...")
    
    for color_name, color_value in dye_colors.items():
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        output_path = f"dye_{color_name}.jpg"
        cv2.imwrite(output_path, colored_image)
        print(f"   ✅ Tạo {output_path} - RGB{color_value}")

def create_color_comparison_grid():
    """Tạo lưới so sánh các màu tóc"""
    print("\n=== TẠO LƯỚI SO SÁNH MÀU ===")
    
    # Tìm ảnh gốc
    test_images = ["test_image.jpg", "test_image.png", "input.jpg"]
    original_path = None
    
    for img in test_images:
        if os.path.exists(img):
            original_path = img
            break
    
    if not original_path:
        print("❌ Không tìm thấy ảnh gốc")
        return
    
    # Tìm các ảnh đã tô màu
    colored_images = []
    
    # Màu tự nhiên
    for color_name in get_natural_colors().keys():
        path = f"natural_{color_name}.jpg"
        if os.path.exists(path):
            colored_images.append((f"Tự nhiên: {color_name}", path))
    
    # Màu nhuộm
    for color_name in get_dye_colors().keys():
        path = f"dye_{color_name}.jpg"
        if os.path.exists(path):
            colored_images.append((f"Nhuộm: {color_name}", path))
    
    # Màu xu hướng
    for color_name in get_trendy_colors().keys():
        path = f"trendy_{color_name}.jpg"
        if os.path.exists(path):
            colored_images.append((f"Xu hướng: {color_name}", path))
    
    if not colored_images:
        print("❌ Không tìm thấy ảnh đã tô màu")
        return
    
    try:
        # Tính số hàng và cột
        total_images = len(colored_images) + 1  # +1 cho ảnh gốc
        cols = 4
        rows = (total_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Hiển thị ảnh gốc
        original_img = cv2.imread(original_path)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Ảnh gốc', fontsize=12)
        axes[0, 0].axis('off')
        
        # Hiển thị các ảnh đã tô màu
        for i, (title, path) in enumerate(colored_images):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            if row < rows and col < cols:
                img = cv2.imread(path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img_rgb)
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis('off')
        
        # Ẩn các subplot trống
        for i in range(total_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig("hair_color_comparison.jpg", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Đã tạo lưới so sánh: hair_color_comparison.jpg")
        
    except Exception as e:
        print(f"❌ Lỗi tạo lưới so sánh: {e}")

def demo_specific_colors():
    """Demo với các màu cụ thể theo yêu cầu"""
    print("\n=== DEMO MÀU CỤ THỂ ===")
    
    segmentator = HairSegmentator()
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists(model_path):
        print("❌ Cần có model để chạy demo")
        return
    
    segmentator.load_model(model_path)
    
    # Tìm ảnh test
    test_images = ["test_image.jpg", "test_image.png", "input.jpg"]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("❌ Cần có ảnh test để chạy demo")
        return
    
    # Phân đoạn tóc một lần
    hair_mask = segmentator.segment_hair(image_path)
    
    # Các màu cụ thể theo yêu cầu
    specific_colors = {
        'Đen tự nhiên': get_hair_color('den_tu_nhien'),
        'Vàng bạch kim': get_hair_color('vang_bach_kim'),
        'Nâu đậm': get_hair_color('nau_dam'),
        'Đỏ cherry': get_hair_color('do_cherry'),
        'Bạc tự nhiên': get_hair_color('bac_tu_nhien'),
        'Vàng mật ong': get_hair_color('vang_mat_ong'),
    }
    
    print("✨ Đang tạo ảnh với màu cụ thể...")
    
    for color_name, color_value in specific_colors.items():
        if color_value:
            colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
            output_path = f"specific_{color_name.replace(' ', '_').lower()}.jpg"
            cv2.imwrite(output_path, colored_image)
            print(f"   ✅ Tạo {output_path} - RGB{color_value}")

def create_color_palette():
    """Tạo bảng màu tóc"""
    print("\n=== TẠO BẢNG MÀU TÓC ===")
    
    try:
        palette = get_color_palette_image()
        cv2.imwrite("hair_color_palette.jpg", palette)
        print("✅ Đã tạo bảng màu: hair_color_palette.jpg")
    except Exception as e:
        print(f"❌ Lỗi tạo bảng màu: {e}")

def main():
    """Hàm chính"""
    print("🎨 DEMO MÀU TÓC THỰC TẾ")
    print("=" * 60)
    
    # Tạo bảng màu trước
    create_color_palette()
    
    # Hiển thị tất cả màu có sẵn
    print("\n📋 DANH SÁCH MÀU CÓ SẴN:")
    list_all_colors()
    
    # Chạy các demo
    demo_natural_hair_colors()
    demo_popular_dye_colors()
    demo_trendy_hair_colors()
    demo_specific_colors()
    
    # Tạo lưới so sánh
    create_color_comparison_grid()
    
    print("\n🎉 HOÀN THÀNH TẤT CẢ DEMO MÀU TÓC!")
    print("Kiểm tra các file kết quả:")
    print("   - hair_color_palette.jpg: Bảng màu tóc")
    print("   - hair_color_comparison.jpg: Lưới so sánh")
    print("   - natural_*.jpg: Màu tóc tự nhiên")
    print("   - dye_*.jpg: Màu tóc nhuộm")
    print("   - trendy_*.jpg: Màu tóc xu hướng")
    print("   - specific_*.jpg: Màu tóc cụ thể")

if __name__ == "__main__":
    main()
