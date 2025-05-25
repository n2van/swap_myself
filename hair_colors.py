"""
Bộ sưu tập màu tóc thực tế cho Hair Segmentation
Chứa các màu tóc phổ biến và tự nhiên
"""

# Màu tóc tự nhiên
NATURAL_HAIR_COLORS = {
    'den_tu_nhien': (30, 30, 30),           # Đen tự nhiên
    'den_nau': (45, 35, 25),                # Đen nâu
    'nau_dam': (101, 67, 33),               # Nâu đậm
    'nau_trung_binh': (139, 117, 79),       # Nâu trung bình
    'nau_sang': (160, 140, 100),            # Nâu sáng
    'nau_vang': (180, 150, 90),             # Nâu vàng
    'vang_dong': (218, 165, 32),            # Vàng đồng
    'vang_mat_ong': (255, 204, 102),        # Vàng mật ong
    'vang_bach_kim': (229, 228, 226),       # Vàng bạch kim
    'bac_tu_nhien': (192, 192, 192),        # Bạc tự nhiên
    'trang_bach_kim': (245, 245, 245),      # Trắng bạch kim
}

# Màu tóc nhuộm phổ biến
POPULAR_DYE_COLORS = {
    'do_cherry': (139, 0, 0),               # Đỏ cherry
    'do_burgundy': (128, 0, 32),            # Đỏ burgundy
    'do_mahogany': (192, 64, 0),            # Đỏ mahogany
    'cam_dong': (255, 140, 0),              # Cam đồng
    'vang_chanh': (255, 255, 0),            # Vàng chanh
    'xanh_la_pastel': (152, 251, 152),      # Xanh lá pastel
    'xanh_duong_navy': (0, 0, 128),         # Xanh dương navy
    'tim_lavender': (230, 230, 250),        # Tím lavender
    'tim_violet': (138, 43, 226),           # Tím violet
    'hong_pastel': (255, 182, 193),         # Hồng pastel
    'hong_rose_gold': (183, 110, 121),      # Hồng rose gold
}

# Màu tóc ombre/gradient
OMBRE_COLORS = {
    'ombre_nau_vang': [(101, 67, 33), (218, 165, 32)],     # Nâu -> Vàng
    'ombre_den_bac': [(30, 30, 30), (192, 192, 192)],      # Đen -> Bạc
    'ombre_nau_do': [(139, 117, 79), (139, 0, 0)],         # Nâu -> Đỏ
    'ombre_vang_hong': [(255, 204, 102), (255, 182, 193)], # Vàng -> Hồng
}

# Màu tóc theo xu hướng
TRENDY_COLORS = {
    'unicorn_pastel': (255, 192, 203),      # Hồng unicorn
    'mermaid_blue': (0, 191, 255),          # Xanh nàng tiên cá
    'galaxy_purple': (75, 0, 130),          # Tím galaxy
    'sunset_orange': (255, 94, 77),         # Cam hoàng hôn
    'mint_green': (152, 255, 152),          # Xanh mint
    'cotton_candy': (255, 183, 197),        # Hồng kẹo bông
    'steel_blue': (70, 130, 180),           # Xanh thép
    'rose_gold': (183, 110, 121),           # Vàng hồng
}

# Tất cả màu tóc
ALL_HAIR_COLORS = {
    **NATURAL_HAIR_COLORS,
    **POPULAR_DYE_COLORS,
    **TRENDY_COLORS
}

def get_hair_color(color_name):
    """
    Lấy màu RGB theo tên
    
    Args:
        color_name: tên màu (str)
    
    Returns:
        tuple RGB hoặc None nếu không tìm thấy
    """
    return ALL_HAIR_COLORS.get(color_name.lower())

def get_natural_colors():
    """Trả về danh sách màu tóc tự nhiên"""
    return NATURAL_HAIR_COLORS

def get_dye_colors():
    """Trả về danh sách màu tóc nhuộm"""
    return POPULAR_DYE_COLORS

def get_trendy_colors():
    """Trả về danh sách màu tóc xu hướng"""
    return TRENDY_COLORS

def list_all_colors():
    """In ra tất cả màu có sẵn"""
    print("=== MÀU TÓC TỰ NHIÊN ===")
    for name, rgb in NATURAL_HAIR_COLORS.items():
        print(f"{name}: {rgb}")
    
    print("\n=== MÀU TÓC NHUỘM PHỔ BIẾN ===")
    for name, rgb in POPULAR_DYE_COLORS.items():
        print(f"{name}: {rgb}")
    
    print("\n=== MÀU TÓC XU HƯỚNG ===")
    for name, rgb in TRENDY_COLORS.items():
        print(f"{name}: {rgb}")

def get_color_palette_image():
    """
    Tạo ảnh palette màu để xem trước
    """
    import cv2
    import numpy as np
    
    # Kích thước mỗi ô màu
    cell_width = 100
    cell_height = 50
    cols = 6
    
    all_colors = list(ALL_HAIR_COLORS.items())
    rows = (len(all_colors) + cols - 1) // cols
    
    # Tạo ảnh trống
    palette = np.ones((rows * cell_height, cols * cell_width, 3), dtype=np.uint8) * 255
    
    for i, (name, color) in enumerate(all_colors):
        row = i // cols
        col = i % cols
        
        # Vẽ ô màu
        y1 = row * cell_height
        y2 = (row + 1) * cell_height
        x1 = col * cell_width
        x2 = (col + 1) * cell_width
        
        # Chuyển RGB sang BGR cho OpenCV
        bgr_color = color[::-1]
        palette[y1:y2, x1:x2] = bgr_color
        
        # Thêm text tên màu (nếu có đủ chỗ)
        if cell_height > 30:
            cv2.putText(palette, name[:8], (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return palette

if __name__ == "__main__":
    # Demo sử dụng
    print("🎨 BỘ SƯU TẬP MÀU TÓC")
    print("=" * 50)
    
    list_all_colors()
    
    # Tạo palette màu
    try:
        palette = get_color_palette_image()
        cv2.imwrite("hair_color_palette.jpg", palette)
        print(f"\n✅ Đã tạo palette màu: hair_color_palette.jpg")
    except ImportError:
        print("\n⚠️ Cần cài opencv-python để tạo palette màu")
    
    # Test lấy màu
    print(f"\nVí dụ lấy màu:")
    print(f"Đen tự nhiên: {get_hair_color('den_tu_nhien')}")
    print(f"Vàng bạch kim: {get_hair_color('vang_bach_kim')}")
    print(f"Đỏ cherry: {get_hair_color('do_cherry')}")
