"""
Demo sử dụng Hair Segmentation
Hướng dẫn cách sử dụng bộ phân đoạn tóc đơn giản
"""

from hair_segmentation_simple import HairSegmentator
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def demo_basic_usage():
    """Demo cách sử dụng cơ bản"""
    print("=== DEMO CƠ BẢN ===")
    
    # 1. Khởi tạo bộ phân đoạn tóc
    segmentator = HairSegmentator()
    
    # 2. Tải model (cần có file model trước)
    model_path = "pretrained/79999_iter.pth"
    if os.path.exists(model_path):
        segmentator.load_model(model_path)
    else:
        print("❌ Chưa có model. Vui lòng tải model từ:")
        print("https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812")
        print("và đặt tại: pretrained/79999_iter.pth")
        return
    
    # 3. Đường dẫn ảnh test
    image_path = "test_image.jpg"  # Thay đổi đường dẫn này
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        print("Vui lòng đặt ảnh test vào thư mục hiện tại")
        return
    
    # 4. Phân đoạn tóc
    print("🔄 Đang phân đoạn tóc...")
    hair_mask = segmentator.segment_hair(image_path)
    
    # 5. Lưu mask
    segmentator.save_mask(hair_mask, "hair_mask_demo.png")
    
    # 6. Tô màu tóc
    colored_image = segmentator.apply_hair_color(image_path, hair_mask, color=(255, 0, 0))
    cv2.imwrite("hair_colored_demo.jpg", colored_image)
    
    print("✅ Hoàn thành! Kiểm tra file:")
    print("   - hair_mask_demo.png (mask tóc)")
    print("   - hair_colored_demo.jpg (tóc màu đỏ)")

def demo_multiple_colors():
    """Demo tô nhiều màu khác nhau"""
    print("\n=== DEMO NHIỀU MÀU ===")
    
    segmentator = HairSegmentator()
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists(model_path):
        print("❌ Cần có model để chạy demo")
        return
    
    segmentator.load_model(model_path)
    
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Cần có ảnh test: {image_path}")
        return
    
    # Phân đoạn tóc một lần
    hair_mask = segmentator.segment_hair(image_path)
    
    # Định nghĩa các màu
    colors = {
        'do': (255, 0, 0),
        'xanh_la': (0, 255, 0),
        'xanh_duong': (0, 0, 255),
        'vang': (255, 255, 0),
        'tim': (255, 0, 255),
        'cam': (255, 165, 0),
        'nau': (165, 42, 42),
        'hong': (255, 192, 203)
    }
    
    print("🎨 Đang tạo ảnh với các màu khác nhau...")
    
    for color_name, color_value in colors.items():
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        output_path = f"hair_{color_name}.jpg"
        cv2.imwrite(output_path, colored_image)
        print(f"   ✅ Tạo {output_path}")

def demo_with_confidence():
    """Demo với độ tin cậy"""
    print("\n=== DEMO VỚI ĐỘ TIN CẬY ===")
    
    segmentator = HairSegmentator()
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists(model_path):
        print("❌ Cần có model để chạy demo")
        return
    
    segmentator.load_model(model_path)
    
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Cần có ảnh test: {image_path}")
        return
    
    # Phân đoạn với độ tin cậy
    hair_mask, confidence = segmentator.segment_hair(image_path, return_confidence=True)
    
    print(f"📊 Độ tin cậy phân đoạn tóc: {confidence:.3f}")
    
    if confidence > 0.8:
        print("✅ Độ tin cậy cao - kết quả tốt")
    elif confidence > 0.5:
        print("⚠️ Độ tin cậy trung bình - có thể cần kiểm tra")
    else:
        print("❌ Độ tin cậy thấp - có thể không có tóc hoặc ảnh không phù hợp")

def create_comparison_image(original_path, mask_path, colored_path, output_path="comparison.jpg"):
    """Tạo ảnh so sánh kết quả"""
    print("\n=== TẠO ẢNH SO SÁNH ===")
    
    try:
        # Đọc các ảnh
        original = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        colored = cv2.imread(colored_path)
        
        # Chuyển BGR sang RGB cho hiển thị
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Tạo figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_rgb)
        axes[0].set_title('Ảnh gốc')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask tóc')
        axes[1].axis('off')
        
        axes[2].imshow(colored_rgb)
        axes[2].set_title('Tóc đã tô màu')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Đã tạo ảnh so sánh: {output_path}")
        
    except Exception as e:
        print(f"❌ Lỗi tạo ảnh so sánh: {e}")

def setup_demo_environment():
    """Thiết lập môi trường demo"""
    print("=== THIẾT LẬP MÔI TRƯỜNG ===")
    
    # Tạo thư mục pretrained nếu chưa có
    os.makedirs("pretrained", exist_ok=True)
    print("✅ Đã tạo thư mục pretrained/")
    
    # Kiểm tra model
    model_path = "pretrained/79999_iter.pth"
    if os.path.exists(model_path):
        print("✅ Model đã có sẵn")
    else:
        print("❌ Chưa có model. Hướng dẫn tải:")
        print("1. Truy cập: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812")
        print("2. Tải file 79999_iter.pth")
        print("3. Đặt vào thư mục pretrained/")
    
    # Kiểm tra ảnh test
    test_images = ["test_image.jpg", "test_image.png", "input.jpg", "input.png"]
    found_image = None
    
    for img in test_images:
        if os.path.exists(img):
            found_image = img
            break
    
    if found_image:
        print(f"✅ Tìm thấy ảnh test: {found_image}")
    else:
        print("❌ Chưa có ảnh test. Vui lòng đặt ảnh với tên:")
        print("   - test_image.jpg hoặc")
        print("   - test_image.png")

def main():
    """Hàm chính chạy tất cả demo"""
    print("🚀 DEMO HAIR SEGMENTATION")
    print("=" * 50)
    
    # Thiết lập môi trường
    setup_demo_environment()
    
    # Chạy các demo
    demo_basic_usage()
    demo_multiple_colors()
    demo_with_confidence()
    
    # Tạo ảnh so sánh nếu có file
    if os.path.exists("test_image.jpg") and os.path.exists("hair_mask_demo.png"):
        create_comparison_image("test_image.jpg", "hair_mask_demo.png", "hair_colored_demo.jpg")
    
    print("\n🎉 HOÀN THÀNH TẤT CẢ DEMO!")
    print("Kiểm tra các file kết quả trong thư mục hiện tại.")

if __name__ == "__main__":
    main()
