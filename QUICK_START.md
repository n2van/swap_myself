# 🚀 Quick Start - Hair Segmentation

Hướng dẫn nhanh để bắt đầu sử dụng Hair Segmentation trong 5 phút!

## ⚡ Cài Đặt Express

```bash
# 1. Cài đặt thư viện
pip install torch torchvision numpy opencv-python pillow matplotlib

# 2. Tạo thư mục model
mkdir pretrained

# 3. Tải model (79999_iter.pth) từ Google Drive vào thư mục pretrained/
```

**📥 Link tải model**: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812

## 🎯 Sử Dụng Ngay

### 1️⃣ Tô màu tóc (3 dòng code)

```python
from hair_segmentation_simple import HairSegmentator

segmentator = HairSegmentator("pretrained/79999_iter.pth")
hair_mask = segmentator.segment_hair("your_photo.jpg")
segmentator.apply_hair_color("your_photo.jpg", hair_mask, (255, 0, 0))  # Đỏ
```

### 2️⃣ Tách tóc ra ngoài (1 dòng code)

```python
segmentator.save_hair_only("your_photo.jpg", hair_mask, "hair.png", "transparent")
```

### 3️⃣ Chạy demo

```bash
# Demo cơ bản
python demo_hair_segmentation.py

# Demo tách tóc
python demo_extract_hair.py

# Demo màu thực tế
python demo_realistic_hair_colors.py
```

## 🎨 Màu Tóc Phổ Biến

```python
# Màu tự nhiên
(30, 30, 30)      # Đen
(101, 67, 33)     # Nâu
(255, 204, 102)   # Vàng

# Màu nhuộm
(255, 0, 0)       # Đỏ
(138, 43, 226)    # Tím
(0, 191, 255)     # Xanh
```

## 📁 File Cần Thiết

```
your_project/
├── hair_segmentation_simple.py  ✅ Code chính
├── your_photo.jpg               ✅ Ảnh của bạn
└── pretrained/
    └── 79999_iter.pth          ✅ Model AI
```

## 🔧 Troubleshooting

**❌ Lỗi "No module named 'hair_segmentation_simple'"**
```bash
# Đảm bảo file hair_segmentation_simple.py ở cùng thư mục
```

**❌ Lỗi "Model not found"**
```bash
# Kiểm tra file pretrained/79999_iter.pth có tồn tại không
```

**❌ Lỗi "Image not found"**
```bash
# Đảm bảo đường dẫn ảnh đúng
```

## 🎉 Kết Quả

Sau khi chạy thành công, bạn sẽ có:
- 🎨 Ảnh tóc đã tô màu
- ✂️ File tóc tách riêng (PNG trong suốt)
- 📊 Mask tóc để xử lý thêm

## 📚 Tài Liệu Đầy Đủ

- 📖 [Hướng dẫn chi tiết](HUONG_DAN_SU_DUNG.md)
- 🎨 [Hướng dẫn màu tóc](HUONG_DAN_MAU_TOC.md)
- 📄 [README đầy đủ](README_VI.md)

---

**🎯 Mục tiêu**: Từ ảnh selfie → Tóc màu mới trong 30 giây!**
