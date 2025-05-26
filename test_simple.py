"""
Test đơn giản để kiểm tra model có load được không
"""

try:
    print("🔧 Đang import module...")
    from hair_segmentation_fixed import HairSegmentator
    print("✅ Import thành công!")
    
    print("🔧 Đang khởi tạo HairSegmentator...")
    segmentator = HairSegmentator()
    print("✅ Khởi tạo thành công!")
    
    print("🔧 Đang test load model...")
    model_path = "pretrained/79999_iter.pth"
    segmentator.load_model(model_path)
    print("✅ Test hoàn thành!")
    
except Exception as e:
    print(f"❌ Lỗi: {e}")
    import traceback
    traceback.print_exc()
