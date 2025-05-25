"""
Script test nhanh cho Hair Segmentation
Chạy để kiểm tra xem code có hoạt động không
"""

import os
import sys
from hair_segmentation_simple import HairSegmentator
import cv2
import numpy as np
from PIL import Image

def create_test_image():
    """Tạo ảnh test đơn giản nếu không có ảnh"""
    print("🎨 Tạo ảnh test đơn giản...")
    
    # Tạo ảnh test với hình tròn đại diện cho khuôn mặt
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Nền trắng
    
    # Vẽ khuôn mặt (hình tròn)
    cv2.circle(img, (200, 220), 80, (255, 220, 177), -1)  # Da mặt
    
    # Vẽ tóc (hình elip)
    cv2.ellipse(img, (200, 160), (100, 60), 0, 0, 180, (139, 69, 19), -1)  # Tóc nâu
    
    # Vẽ mắt
    cv2.circle(img, (180, 200), 8, (0, 0, 0), -1)  # Mắt trái
    cv2.circle(img, (220, 200), 8, (0, 0, 0), -1)  # Mắt phải
    
    # Vẽ miệng
    cv2.ellipse(img, (200, 240), (15, 8), 0, 0, 180, (255, 0, 0), 2)  # Miệng
    
    cv2.imwrite("test_face.jpg", img)
    print("✅ Đã tạo ảnh test: test_face.jpg")
    return "test_face.jpg"

def check_requirements():
    """Kiểm tra các yêu cầu cần thiết"""
    print("🔍 Kiểm tra yêu cầu...")
    
    # Kiểm tra thư viện
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        from PIL import Image
        print("✅ Tất cả thư viện đã được cài đặt")
    except ImportError as e:
        print(f"❌ Thiếu thư viện: {e}")
        print("Chạy: pip install torch torchvision opencv-python pillow numpy")
        return False
    
    # Kiểm tra CUDA
    if torch.cuda.is_available():
        print(f"✅ GPU có sẵn: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Chỉ có CPU (vẫn có thể chạy được)")
    
    return True

def test_model_loading():
    """Test tải model"""
    print("\n📦 Test tải model...")
    
    model_path = "pretrained/79999_iter.pth"
    
    if not os.path.exists("pretrained"):
        os.makedirs("pretrained")
        print("✅ Đã tạo thư mục pretrained/")
    
    if not os.path.exists(model_path):
        print("❌ Chưa có model. Cần tải từ:")
        print("https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812")
        print("và đặt tại: pretrained/79999_iter.pth")
        return False
    
    try:
        segmentator = HairSegmentator()
        segmentator.load_model(model_path)
        print("✅ Model tải thành công")
        return True
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return False

def test_segmentation():
    """Test phân đoạn tóc"""
    print("\n✂️ Test phân đoạn tóc...")
    
    # Tìm ảnh test
    test_images = ["test_image.jpg", "test_face.jpg", "input.jpg"]
    image_path = None
    
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        # Tạo ảnh test
        image_path = create_test_image()
    
    print(f"📸 Sử dụng ảnh: {image_path}")
    
    try:
        # Khởi tạo segmentator
        segmentator = HairSegmentator()
        
        model_path = "pretrained/79999_iter.pth"
        if os.path.exists(model_path):
            segmentator.load_model(model_path)
        else:
            print("❌ Cần có model để test phân đoạn")
            return False
        
        # Phân đoạn tóc
        print("🔄 Đang phân đoạn...")
        hair_mask = segmentator.segment_hair(image_path)
        
        # Lưu mask
        segmentator.save_mask(hair_mask, "test_mask.png")
        
        # Tô màu đỏ
        colored = segmentator.apply_hair_color(image_path, hair_mask, (255, 0, 0))
        cv2.imwrite("test_colored.jpg", colored)
        
        print("✅ Test phân đoạn thành công!")
        print("   - test_mask.png: mask tóc")
        print("   - test_colored.jpg: tóc màu đỏ")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi phân đoạn: {e}")
        return False

def test_multiple_colors():
    """Test nhiều màu"""
    print("\n🌈 Test nhiều màu...")
    
    if not os.path.exists("test_mask.png"):
        print("❌ Cần chạy test phân đoạn trước")
        return False
    
    try:
        segmentator = HairSegmentator("pretrained/79999_iter.pth")
        
        # Đọc mask đã tạo
        hair_mask = cv2.imread("test_mask.png", cv2.IMREAD_GRAYSCALE)
        
        # Test các màu
        colors = {
            'xanh': (0, 255, 0),
            'tim': (255, 0, 255),
            'vang': (255, 255, 0)
        }
        
        image_path = "test_face.jpg" if os.path.exists("test_face.jpg") else "test_image.jpg"
        
        for color_name, color_value in colors.items():
            colored = segmentator.apply_hair_color(image_path, hair_mask, color_value)
            output_path = f"test_{color_name}.jpg"
            cv2.imwrite(output_path, colored)
            print(f"   ✅ Tạo {output_path}")
        
        print("✅ Test nhiều màu thành công!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test màu: {e}")
        return False

def cleanup_test_files():
    """Dọn dẹp file test"""
    test_files = [
        "test_face.jpg", "test_mask.png", "test_colored.jpg",
        "test_xanh.jpg", "test_tim.jpg", "test_vang.jpg"
    ]
    
    print("\n🧹 Dọn dẹp file test...")
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   🗑️ Đã xóa {file}")
            except:
                print(f"   ⚠️ Không thể xóa {file}")

def main():
    """Hàm chính"""
    print("🚀 HAIR SEGMENTATION - TEST NHANH")
    print("=" * 50)
    
    # Kiểm tra yêu cầu
    if not check_requirements():
        print("\n❌ Không đủ yêu cầu để chạy test")
        return
    
    # Test tải model
    if not test_model_loading():
        print("\n❌ Không thể tải model")
        print("Vui lòng tải model trước khi test")
        return
    
    # Test phân đoạn
    if not test_segmentation():
        print("\n❌ Test phân đoạn thất bại")
        return
    
    # Test nhiều màu
    test_multiple_colors()
    
    print("\n🎉 TẤT CẢ TEST HOÀN THÀNH!")
    print("Code phân đoạn tóc hoạt động bình thường.")
    
    # Hỏi có muốn dọn dẹp không
    try:
        response = input("\nBạn có muốn xóa file test không? (y/n): ").lower()
        if response in ['y', 'yes', 'có']:
            cleanup_test_files()
    except:
        pass

if __name__ == "__main__":
    main()
