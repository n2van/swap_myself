# Hướng Dẫn Màu Tóc Thực Tế

## Giới thiệu

Bộ sưu tập màu tóc thực tế cho Hair Segmentation với hơn 30 màu tóc tự nhiên và xu hướng.

## Các loại màu tóc

### 1. Màu tóc tự nhiên

| Tên màu | RGB | Mô tả |
|---------|-----|-------|
| Đen tự nhiên | (30, 30, 30) | Màu đen tóc châu Á truyền thống |
| Đen nâu | (45, 35, 25) | Đen pha chút nâu |
| Nâu đậm | (101, 67, 33) | Nâu chocolate đậm |
| Nâu trung bình | (139, 117, 79) | Nâu cà phê sữa |
| Nâu sáng | (160, 140, 100) | Nâu hạt dẻ |
| Nâu vàng | (180, 150, 90) | Nâu pha vàng |
| Vàng đồng | (218, 165, 32) | Vàng kim loại |
| Vàng mật ong | (255, 204, 102) | Vàng ngọt ngào |
| Vàng bạch kim | (229, 228, 226) | Vàng platinum |
| Bạc tự nhiên | (192, 192, 192) | Bạc tóc già |
| Trắng bạch kim | (245, 245, 245) | Trắng như tuyết |

### 2. Màu tóc nhuộm phổ biến

| Tên màu | RGB | Mô tả |
|---------|-----|-------|
| Đỏ cherry | (139, 0, 0) | Đỏ anh đào |
| Đỏ burgundy | (128, 0, 32) | Đỏ rượu vang |
| Đỏ mahogany | (192, 64, 0) | Đỏ gỗ gụ |
| Cam đồng | (255, 140, 0) | Cam kim loại |
| Vàng chanh | (255, 255, 0) | Vàng neon |
| Xanh lá pastel | (152, 251, 152) | Xanh nhạt |
| Xanh dương navy | (0, 0, 128) | Xanh hải quân |
| Tím lavender | (230, 230, 250) | Tím oải hương |
| Tím violet | (138, 43, 226) | Tím đậm |
| Hồng pastel | (255, 182, 193) | Hồng nhạt |
| Hồng rose gold | (183, 110, 121) | Hồng vàng |

### 3. Màu tóc xu hướng

| Tên màu | RGB | Mô tả |
|---------|-----|-------|
| Unicorn pastel | (255, 192, 203) | Hồng kỳ lân |
| Mermaid blue | (0, 191, 255) | Xanh nàng tiên cá |
| Galaxy purple | (75, 0, 130) | Tím thiên hà |
| Sunset orange | (255, 94, 77) | Cam hoàng hôn |
| Mint green | (152, 255, 152) | Xanh bạc hà |
| Cotton candy | (255, 183, 197) | Hồng kẹo bông |
| Steel blue | (70, 130, 180) | Xanh thép |
| Rose gold | (183, 110, 121) | Vàng hồng |

## Cách sử dụng

### 1. Sử dụng với hair_colors.py

```python
from hair_colors import get_hair_color, get_natural_colors
from hair_segmentation_simple import HairSegmentator

# Khởi tạo
segmentator = HairSegmentator("pretrained/79999_iter.pth")

# Lấy màu cụ thể
black_color = get_hair_color('den_tu_nhien')
platinum_color = get_hair_color('vang_bach_kim')

# Phân đoạn và tô màu
hair_mask = segmentator.segment_hair("anh.jpg")
colored_image = segmentator.apply_hair_color("anh.jpg", hair_mask, black_color)
```

### 2. Chạy demo màu thực tế

```bash
python demo_realistic_hair_colors.py
```

Demo sẽ tạo:
- Bảng màu tóc (hair_color_palette.jpg)
- Ảnh với màu tự nhiên (natural_*.jpg)
- Ảnh với màu nhuộm (dye_*.jpg)
- Ảnh với màu xu hướng (trendy_*.jpg)
- Lưới so sánh tất cả màu

### 3. Tạo bảng màu

```python
from hair_colors import get_color_palette_image
import cv2

# Tạo bảng màu
palette = get_color_palette_image()
cv2.imwrite("bang_mau_toc.jpg", palette)
```

## Ví dụ sử dụng nâng cao

### 1. Thử nhiều màu tự nhiên

```python
from hair_colors import get_natural_colors
from hair_segmentation_simple import HairSegmentator

segmentator = HairSegmentator("pretrained/79999_iter.pth")
hair_mask = segmentator.segment_hair("anh.jpg")

# Thử tất cả màu tự nhiên
natural_colors = get_natural_colors()
for color_name, color_rgb in natural_colors.items():
    colored = segmentator.apply_hair_color("anh.jpg", hair_mask, color_rgb)
    cv2.imwrite(f"toc_{color_name}.jpg", colored)
```

### 2. So sánh màu trước và sau

```python
import matplotlib.pyplot as plt

# Đọc ảnh gốc và ảnh đã tô màu
original = cv2.imread("anh.jpg")
colored = cv2.imread("toc_vang_bach_kim.jpg")

# Hiển thị so sánh
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title('Ảnh gốc')
axes[1].imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
axes[1].set_title('Tóc vàng bạch kim')
plt.show()
```

### 3. Tùy chỉnh độ đậm màu

```python
# Màu nhẹ (blend_ratio thấp)
light_colored = segmentator.apply_hair_color(
    "anh.jpg", hair_mask, 
    get_hair_color('do_cherry'), 
    blend_ratio=0.3
)

# Màu đậm (blend_ratio cao)
dark_colored = segmentator.apply_hair_color(
    "anh.jpg", hair_mask, 
    get_hair_color('do_cherry'), 
    blend_ratio=0.9
)
```

## Màu tóc theo độ tuổi

### Trẻ em (0-12 tuổi)
- Đen tự nhiên
- Nâu đậm
- Nâu trung bình

### Thanh thiếu niên (13-19 tuổi)
- Nâu vàng
- Vàng mật ong
- Đỏ cherry (nhẹ)

### Người trẻ (20-35 tuổi)
- Tất cả màu xu hướng
- Màu nhuộm phổ biến
- Ombre/gradient

### Trung niên (36-55 tuổi)
- Màu tự nhiên
- Nâu sang trọng
- Đỏ burgundy

### Cao tuổi (55+ tuổi)
- Bạc tự nhiên
- Trắng bạch kim
- Nâu nhạt

## Màu tóc theo tông da

### Da sáng
- Vàng bạch kim
- Trắng bạch kim
- Hồng pastel
- Tím lavender

### Da trung bình
- Nâu trung bình
- Vàng mật ong
- Đỏ cherry
- Cam đồng

### Da ngăm
- Đen tự nhiên
- Nâu đậm
- Đỏ mahogany
- Tím violet

## Tips sử dụng

### 1. Chọn màu phù hợp
- Xem xét tông da
- Phù hợp với độ tuổi
- Môi trường làm việc

### 2. Điều chỉnh độ tự nhiên
- blend_ratio = 0.3-0.5: Tự nhiên
- blend_ratio = 0.6-0.8: Nổi bật
- blend_ratio = 0.9-1.0: Dramatic

### 3. Kết hợp màu
- Ombre: Từ tối đến sáng
- Highlights: Điểm nhấn sáng màu
- Lowlights: Điểm nhấn tối màu

## Troubleshooting

### Màu không hiển thị đúng
- Kiểm tra mask tóc có chính xác không
- Thử điều chỉnh blend_ratio
- Đảm bảo ảnh đầu vào chất lượng tốt

### Màu quá đậm/nhạt
- Điều chỉnh blend_ratio
- Thử màu khác trong cùng nhóm
- Kiểm tra độ sáng ảnh gốc

### Kết quả không tự nhiên
- Sử dụng màu từ nhóm "tự nhiên"
- Giảm blend_ratio xuống 0.5
- Thử với ảnh có ánh sáng tốt hơn

## File liên quan

- `hair_colors.py`: Định nghĩa tất cả màu
- `demo_realistic_hair_colors.py`: Demo màu thực tế
- `hair_segmentation_simple.py`: Code chính
- `HUONG_DAN_SU_DUNG.md`: Hướng dẫn tổng quát
