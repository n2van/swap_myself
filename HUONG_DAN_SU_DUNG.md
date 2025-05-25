# Hướng Dẫn Sử Dụng Hair Segmentation

## Giới thiệu

Dự án này cung cấp công cụ phân đoạn tóc từ ảnh khuôn mặt và tô màu tóc tự động. Dựa trên repository [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) với mạng BiSeNet.

## Cài đặt

### 1. Cài đặt thư viện cần thiết

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib
```

### 2. Tải model đã huấn luyện

1. Tạo thư mục `pretrained`:
   ```bash
   mkdir pretrained
   ```

2. Tải model từ Google Drive:
   - Link: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
   - Lưu file với tên: `pretrained/79999_iter.pth`

## Cách sử dụng

### 1. Sử dụng cơ bản

```python
from hair_segmentation_simple import HairSegmentator

# Khởi tạo
segmentator = HairSegmentator()

# Tải model
segmentator.load_model("pretrained/79999_iter.pth")

# Phân đoạn tóc
hair_mask = segmentator.segment_hair("anh_cua_ban.jpg")

# Lưu mask
segmentator.save_mask(hair_mask, "mask_toc.png")

# Tô màu tóc
colored_image = segmentator.apply_hair_color("anh_cua_ban.jpg", hair_mask, color=(255, 0, 0))
cv2.imwrite("toc_mau_do.jpg", colored_image)
```

### 2. Chạy demo

```bash
python demo_hair_segmentation.py
```

Demo sẽ:
- Kiểm tra môi trường
- Phân đoạn tóc từ ảnh test
- Tạo nhiều màu tóc khác nhau
- Hiển thị độ tin cậy
- Tạo ảnh so sánh kết quả

### 3. Sử dụng script có sẵn

```bash
# Chỉ tạo mask tóc
python hair_segmentation.py --input anh.jpg --model pretrained/79999_iter.pth --output mask.png

# Tạo mask và tô màu
python hair_segmentation.py --input anh.jpg --model pretrained/79999_iter.pth --output mask.png --color 255,0,0 --colored_output toc_do.jpg
```

## Các tính năng

### 1. Phân đoạn tóc
- Tự động nhận diện vùng tóc trong ảnh
- Tạo mask nhị phân (trắng = tóc, đen = không phải tóc)
- Hỗ trợ nhiều kích thước ảnh

### 2. Tô màu tóc
- Áp dụng màu lên vùng tóc đã phân đoạn
- Điều chỉnh độ trong suốt
- Nhiều màu có sẵn

### 3. Đánh giá độ tin cậy
- Tính toán độ tin cậy của việc phân đoạn
- Giúp đánh giá chất lượng kết quả

## Màu sắc có sẵn

| Màu | RGB |
|-----|-----|
| Đỏ | (255, 0, 0) |
| Xanh lá | (0, 255, 0) |
| Xanh dương | (0, 0, 255) |
| Vàng | (255, 255, 0) |
| Tím | (255, 0, 255) |
| Cam | (255, 165, 0) |
| Nâu | (165, 42, 42) |
| Hồng | (255, 192, 203) |

## Cấu trúc file

```
├── hair_segmentation_simple.py    # Code chính
├── demo_hair_segmentation.py      # Demo sử dụng
├── hair_segmentation.py           # Script command line
├── hair_segmentation.ipynb        # Jupyter notebook
├── HUONG_DAN_SU_DUNG.md          # Hướng dẫn này
├── README.md                      # README tiếng Anh
└── pretrained/
    └── 79999_iter.pth            # Model đã huấn luyện
```

## Ví dụ sử dụng nâng cao

### 1. Xử lý nhiều ảnh

```python
import os
from hair_segmentation_simple import HairSegmentator

segmentator = HairSegmentator("pretrained/79999_iter.pth")

# Xử lý tất cả ảnh trong thư mục
input_folder = "input_images"
output_folder = "output_images"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        
        # Phân đoạn tóc
        hair_mask = segmentator.segment_hair(image_path)
        
        # Tô màu
        colored = segmentator.apply_hair_color(image_path, hair_mask, (255, 0, 0))
        
        # Lưu kết quả
        output_path = os.path.join(output_folder, f"colored_{filename}")
        cv2.imwrite(output_path, colored)
```

### 2. Điều chỉnh tham số

```python
# Thay đổi tỷ lệ pha trộn màu
colored = segmentator.apply_hair_color(
    image_path, 
    hair_mask, 
    color=(255, 0, 0), 
    blend_ratio=0.5  # Giảm độ đậm màu
)

# Kiểm tra độ tin cậy
hair_mask, confidence = segmentator.segment_hair(image_path, return_confidence=True)
if confidence > 0.8:
    print("Kết quả tốt!")
else:
    print("Cần kiểm tra lại")
```

## Xử lý lỗi thường gặp

### 1. Lỗi không tìm thấy model
```
FileNotFoundError: Không tìm thấy file model
```
**Giải pháp**: Tải model từ Google Drive và đặt đúng vị trí

### 2. Lỗi CUDA
```
RuntimeError: CUDA out of memory
```
**Giải pháp**: Code tự động chuyển sang CPU nếu không có GPU

### 3. Lỗi kích thước ảnh
```
ValueError: Input image size not supported
```
**Giải pháp**: Code tự động resize ảnh về 512x512

### 4. Kết quả không chính xác
- Kiểm tra ảnh đầu vào có rõ nét không
- Đảm bảo khuôn mặt trong ảnh rõ ràng
- Thử với ảnh có độ phân giải cao hơn

## Tối ưu hóa hiệu suất

### 1. Sử dụng GPU
```python
# Kiểm tra GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 2. Xử lý batch
```python
# Xử lý nhiều ảnh cùng lúc (cần modify code)
# Hiện tại code xử lý từng ảnh một
```

## Liên hệ và hỗ trợ

- Repository gốc: https://github.com/zllrunning/face-parsing.PyTorch
- Model: CelebAMask-HQ dataset
- Kiến trúc: BiSeNet với ResNet18 backbone

## Ghi chú

- Model được huấn luyện trên dataset CelebAMask-HQ
- Hoạt động tốt nhất với ảnh khuôn mặt rõ nét
- Hỗ trợ 19 class phân đoạn (tóc là class 17)
- Kích thước đầu vào được resize về 512x512 để xử lý
