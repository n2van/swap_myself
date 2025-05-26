"""
Test script để kiểm tra model có load được không
"""

from hair_segmentation_simple import HairSegmentator
import torch

def test_model_loading():
    """Test xem model có load được không"""
    print("🔧 Đang test model loading...")
    
    try:
        # Khởi tạo
        segmentator = HairSegmentator()
        
        # Test load model
        model_path = "pretrained/79999_iter.pth"
        segmentator.load_model(model_path)
        
        print("✅ Model đã load thành công!")
        print(f"Device: {segmentator.device}")
        print(f"Model parameters: {sum(p.numel() for p in segmentator.model.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def test_simple_inference():
    """Test inference đơn giản"""
    print("\n🧪 Đang test inference...")
    
    try:
        segmentator = HairSegmentator()
        segmentator.load_model("pretrained/79999_iter.pth")
        
        # Tạo tensor test
        test_input = torch.randn(1, 3, 512, 512).to(segmentator.device)
        
        with torch.no_grad():
            outputs = segmentator.model(test_input)
            print(f"✅ Inference thành công!")
            print(f"Output shape: {outputs[0].shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Lỗi inference: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING MODEL COMPATIBILITY")
    print("=" * 50)
    
    # Test 1: Model loading
    success1 = test_model_loading()
    
    # Test 2: Simple inference
    if success1:
        success2 = test_simple_inference()
    else:
        success2 = False
    
    print("\n📊 KẾT QUẢ:")
    print(f"Model Loading: {'✅' if success1 else '❌'}")
    print(f"Inference: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 Model hoạt động bình thường!")
        print("Bạn có thể sử dụng:")
        print("- python demo_hair_segmentation.py")
        print("- python demo_extract_hair.py")
    else:
        print("\n⚠️ Cần sửa model architecture")
