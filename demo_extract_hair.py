"""
Demo tách tóc ra ngoài thành ảnh riêng biệt
Hướng dẫn cách sử dụng tính năng extract_hair_only
"""

from hair_segmentation_simple import HairSegmentator
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def demo_extract_hair():
    """Demo tách tóc với các loại nền khác nhau"""
    print("=== DEMO TÁCH TÓC ===")
    
    # 1. Khởi tạo bộ phân đoạn tóc
    segmentator = HairSegmentator()
    
    # 2. Tải model
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
    
    # 5. Tách tóc với các loại nền khác nhau
    print("✂️ Đang tách tóc với các loại nền...")
    
    # Nền trong suốt (PNG)
    segmentator.save_hair_only(image_path, hair_mask, "hair_transparent.png", "transparent")
    
    # Nền trắng
    segmentator.save_hair_only(image_path, hair_mask, "hair_white_bg.jpg", "white")
    
    # Nền đen
    segmentator.save_hair_only(image_path, hair_mask, "hair_black_bg.jpg", "black")
    
    print("✅ Hoàn thành! Đã tạo các file:")
    print("   - hair_transparent.png (nền trong suốt)")
    print("   - hair_white_bg.jpg (nền trắng)")
    print("   - hair_black_bg.jpg (nền đen)")

def demo_extract_and_color():
    """Demo tách tóc và tô màu"""
    print("\n=== DEMO TÁCH TÓC VÀ TÔ MÀU ===")
    
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
    
    # Phân đoạn tóc
    hair_mask = segmentator.segment_hair(image_path)
    
    # Các màu để test
    colors = {
        'do': (255, 0, 0),
        'xanh': (0, 0, 255),
        'tim': (255, 0, 255),
        'vang': (255, 255, 0)
    }
    
    print("🎨 Đang tạo tóc tô màu với nền trong suốt...")
    
    for color_name, color_value in colors.items():
        # Tô màu tóc trước
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        
        # Tách tóc đã tô màu với nền trong suốt
        hair_only = segmentator.extract_hair_only(image_path, hair_mask, "transparent")
        
        # Tô màu lên phần tóc đã tách
        mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
        colored_hair = hair_only.copy()
        colored_hair[:, :, :3] = colored_hair[:, :, :3] * (1 - mask_3d * 0.7) + np.array(color_value[::-1]) * mask_3d * 0.7
        
        # Lưu file
        output_path = f"hair_colored_{color_name}_transparent.png"
        hair_rgba = cv2.cvtColor(colored_hair, cv2.COLOR_BGRA2RGBA)
        pil_image = Image.fromarray(hair_rgba, 'RGBA')
        pil_image.save(output_path)
        
        print(f"   ✅ Tạo {output_path}")

def create_hair_comparison():
    """Tạo ảnh so sánh các loại tách tóc"""
    print("\n=== TẠO ẢNH SO SÁNH ===")
    
    try:
        # Đọc ảnh gốc và các ảnh đã tách
        original = cv2.imread("test_image.jpg")
        hair_white = cv2.imread("hair_white_bg.jpg")
        hair_black = cv2.imread("hair_black_bg.jpg")
        
        # Đọc ảnh PNG với nền trong suốt
        hair_transparent = cv2.imread("hair_transparent.png", cv2.IMREAD_UNCHANGED)
        
        if hair_transparent.shape[2] == 4:  # Có alpha channel
            # Tạo nền caro để hiển thị độ trong suốt
            h, w = hair_transparent.shape[:2]
            checker = np.zeros((h, w, 3), dtype=np.uint8)
            checker[::20, ::20] = 200
            checker[10::20, 10::20] = 200
            
            # Blend với nền caro
            alpha = hair_transparent[:, :, 3:4] / 255.0
            hair_display = hair_transparent[:, :, :3] * alpha + checker * (1 - alpha)
            hair_display = hair_display.astype(np.uint8)
        else:
            hair_display = hair_transparent[:, :, :3]
        
        # Chuyển BGR sang RGB cho matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        hair_white_rgb = cv2.cvtColor(hair_white, cv2.COLOR_BGR2RGB)
        hair_black_rgb = cv2.cvtColor(hair_black, cv2.COLOR_BGR2RGB)
        hair_display_rgb = cv2.cvtColor(hair_display, cv2.COLOR_BGR2RGB)
        
        # Tạo figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Ảnh gốc')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(hair_display_rgb)
        axes[0, 1].set_title('Tóc - nền trong suốt')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(hair_white_rgb)
        axes[1, 0].set_title('Tóc - nền trắng')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(hair_black_rgb)
        axes[1, 1].set_title('Tóc - nền đen')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("hair_extraction_comparison.jpg", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Đã tạo ảnh so sánh: hair_extraction_comparison.jpg")
        
    except Exception as e:
        print(f"❌ Lỗi tạo ảnh so sánh: {e}")

def demo_hair_on_new_background():
    """Demo đặt tóc lên nền mới"""
    print("\n=== DEMO ĐẶT TÓC LÊN NỀN MỚI ===")
    
    try:
        # Đọc tóc với nền trong suốt
        hair_png = Image.open("hair_transparent.png").convert("RGBA")
        
        # Tạo các nền mới
        w, h = hair_png.size
        
        # Nền gradient
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            gradient[i, :] = [int(255 * i / h), int(128 * (1 - i / h)), 200]
        
        # Nền màu sắc
        backgrounds = {
            'gradient': gradient,
            'blue': np.full((h, w, 3), [100, 150, 255], dtype=np.uint8),
            'pink': np.full((h, w, 3), [255, 200, 220], dtype=np.uint8),
            'green': np.full((h, w, 3), [150, 255, 150], dtype=np.uint8)
        }
        
        for bg_name, bg_color in backgrounds.items():
            # Tạo nền
            background = Image.fromarray(bg_color, 'RGB').convert('RGBA')
            
            # Composite tóc lên nền
            result = Image.alpha_composite(background, hair_png)
            
            # Lưu file
            output_path = f"hair_on_{bg_name}_background.png"
            result.save(output_path)
            print(f"   ✅ Tạo {output_path}")
            
    except Exception as e:
        print(f"❌ Lỗi tạo nền mới: {e}")

def main():
    """Hàm chính chạy tất cả demo"""
    print("✂️ DEMO TÁCH TÓC RA NGOÀI")
    print("=" * 50)
    
    # Chạy các demo
    demo_extract_hair()
    demo_extract_and_color()
    create_hair_comparison()
    demo_hair_on_new_background()
    
    print("\n🎉 HOÀN THÀNH TẤT CẢ DEMO!")
    print("\n📁 Các file đã tạo:")
    print("   - hair_transparent.png (tóc nền trong suốt)")
    print("   - hair_white_bg.jpg (tóc nền trắng)")
    print("   - hair_black_bg.jpg (tóc nền đen)")
    print("   - hair_colored_*_transparent.png (tóc tô màu)")
    print("   - hair_extraction_comparison.jpg (ảnh so sánh)")
    print("   - hair_on_*_background.png (tóc trên nền mới)")
    
    print("\n💡 Cách sử dụng:")
    print("1. Dùng file PNG trong suốt để ghép vào ảnh khác")
    print("2. Dùng nền trắng/đen để in ấn hoặc thiết kế")
    print("3. Có thể tô màu trước khi tách để có tóc màu riêng")

if __name__ == "__main__":
    main()
