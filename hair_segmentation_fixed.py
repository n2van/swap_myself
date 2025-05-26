import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os
import torchvision.models as models

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feat8 = self.resnet.layer1(x)
        feat16 = self.resnet.layer2(feat8)
        feat32 = self.resnet.layer3(feat16)
        feat32 = self.resnet.layer4(feat32)

        avg = nn.functional.adaptive_avg_pool2d(feat32, (1, 1))
        avg = self.conv_avg(avg)
        avg_up = nn.functional.interpolate(avg, size=feat32.size()[2:], mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = nn.functional.interpolate(feat32_sum, size=feat16.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = nn.functional.interpolate(feat16_sum, size=feat8.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = nn.functional.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = nn.functional.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = nn.functional.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

class HairSegmentator:
    def __init__(self, model_path=None):
        """
        Khởi tạo bộ phân đoạn tóc
        
        Args:
            model_path: đường dẫn đến file model đã huấn luyện (.pth)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BiSeNet(n_classes=19)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Đã tải model từ: {model_path}")
            except Exception as e:
                print(f"❌ Lỗi tải model: {e}")
        else:
            print("⚠️ Cảnh báo: Chưa tải model. Vui lòng tải model bằng load_model()")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Chuẩn hóa ảnh theo ImageNet
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def load_model(self, model_path):
        """Tải model từ file"""
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Đã tải model từ: {model_path}")
            except Exception as e:
                print(f"❌ Lỗi tải model: {e}")
                raise e
        else:
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    
    def preprocess_image(self, image_path, target_size=512):
        """
        Tiền xử lý ảnh đầu vào
        
        Args:
            image_path: đường dẫn đến ảnh hoặc PIL Image
            target_size: kích thước ảnh sau khi resize
        
        Returns:
            tensor ảnh đã được tiền xử lý
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Lưu kích thước gốc
        original_size = image.size
        
        # Resize ảnh
        image = image.resize((target_size, target_size), Image.BILINEAR)
        
        # Chuyển đổi thành tensor
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor.to(self.device), original_size
    
    def segment_hair(self, image_path, return_confidence=False):
        """
        Phân đoạn tóc từ ảnh
        
        Args:
            image_path: đường dẫn đến ảnh hoặc PIL Image
            return_confidence: có trả về độ tin cậy không
        
        Returns:
            hair_mask: mask tóc (numpy array)
            confidence: độ tin cậy (nếu return_confidence=True)
        """
        # Tiền xử lý ảnh
        image_tensor, original_size = self.preprocess_image(image_path)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = outputs[0].squeeze(0).cpu().numpy()
        
        # Lấy class có xác suất cao nhất
        segmentation = np.argmax(prediction, axis=0)
        
        # Tạo mask tóc (class 17 là tóc trong CelebAMask-HQ)
        hair_mask = (segmentation == 17).astype(np.uint8) * 255
        
        # Resize về kích thước gốc
        hair_mask = cv2.resize(hair_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        if return_confidence:
            # Tính độ tin cậy trung bình cho vùng tóc
            hair_confidence = np.max(prediction[17]) if np.any(segmentation == 17) else 0.0
            return hair_mask, hair_confidence
        
        return hair_mask
    
    def save_mask(self, mask, output_path):
        """Lưu mask ra file"""
        cv2.imwrite(output_path, mask)
        print(f"Đã lưu mask tóc tại: {output_path}")
    
    def apply_hair_color(self, image_path, hair_mask, color=(255, 0, 0), blend_ratio=0.7):
        """
        Tô màu tóc
        
        Args:
            image_path: đường dẫn đến ảnh gốc
            hair_mask: mask tóc
            color: màu RGB (R, G, B)
            blend_ratio: tỷ lệ pha trộn màu (0-1)
        
        Returns:
            ảnh đã tô màu tóc
        """
        # Đọc ảnh gốc
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        # Đảm bảo mask có cùng kích thước với ảnh
        if hair_mask.shape[:2] != image.shape[:2]:
            hair_mask = cv2.resize(hair_mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Tạo overlay màu
        color_overlay = np.zeros_like(image)
        color_overlay[:] = color[::-1]  # Chuyển RGB sang BGR
        
        # Áp dụng màu lên vùng tóc
        mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
        colored_image = image * (1 - mask_3d * blend_ratio) + color_overlay * mask_3d * blend_ratio
        
        return colored_image.astype(np.uint8)
    
    def extract_hair_only(self, image_path, hair_mask, background_type='transparent'):
        """
        Tách phần tóc ra thành ảnh riêng biệt
        
        Args:
            image_path: đường dẫn đến ảnh gốc
            hair_mask: mask tóc
            background_type: loại nền ('transparent', 'white', 'black')
        
        Returns:
            ảnh chỉ có phần tóc
        """
        # Đọc ảnh gốc
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        # Đảm bảo mask có cùng kích thước với ảnh
        if hair_mask.shape[:2] != image.shape[:2]:
            hair_mask = cv2.resize(hair_mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Tạo mask 3 kênh
        mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        if background_type == 'transparent':
            # Tạo ảnh RGBA (có alpha channel)
            hair_only = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            
            # Copy phần tóc
            hair_only[:, :, :3] = image * mask_3d
            
            # Tạo alpha channel (độ trong suốt)
            hair_only[:, :, 3] = hair_mask
            
            return hair_only
            
        elif background_type == 'white':
            # Nền trắng
            white_bg = np.ones_like(image) * 255
            hair_only = image * mask_3d + white_bg * (1 - mask_3d)
            
        elif background_type == 'black':
            # Nền đen
            black_bg = np.zeros_like(image)
            hair_only = image * mask_3d + black_bg * (1 - mask_3d)
            
        else:
            raise ValueError("background_type phải là 'transparent', 'white', hoặc 'black'")
        
        return hair_only.astype(np.uint8)
    
    def save_hair_only(self, image_path, hair_mask, output_path, background_type='transparent'):
        """
        Tách và lưu phần tóc ra file riêng
        
        Args:
            image_path: đường dẫn đến ảnh gốc
            hair_mask: mask tóc
            output_path: đường dẫn file đầu ra
            background_type: loại nền ('transparent', 'white', 'black')
        """
        hair_only = self.extract_hair_only(image_path, hair_mask, background_type)
        
        if background_type == 'transparent':
            # Lưu dưới dạng PNG để giữ alpha channel
            if not output_path.lower().endswith('.png'):
                output_path = output_path.rsplit('.', 1)[0] + '.png'
            
            # Chuyển BGRA sang RGBA cho PIL
            hair_rgba = cv2.cvtColor(hair_only, cv2.COLOR_BGRA2RGBA)
            pil_image = Image.fromarray(hair_rgba, 'RGBA')
            pil_image.save(output_path)
        else:
            cv2.imwrite(output_path, hair_only)
        
        print(f"Đã tách và lưu phần tóc tại: {output_path}")

def main():
    """Hàm demo sử dụng"""
    print("🎨 Hair Segmentation Demo")
    print("=" * 40)
    
    # Khởi tạo bộ phân đoạn tóc
    segmentator = HairSegmentator()
    
    # Tải model (cần tải model trước)
    model_path = "pretrained/79999_iter.pth"
    if os.path.exists(model_path):
        segmentator.load_model(model_path)
    else:
        print(f"❌ Vui lòng tải model và đặt tại: {model_path}")
        print("📥 Link tải: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812")
        return
    
    # Đường dẫn ảnh đầu vào
    image_path = "input_image.jpg"  # Thay đổi đường dẫn này
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        print("💡 Vui lòng đặt ảnh test với tên 'input_image.jpg'")
        return
    
    # Phân đoạn tóc
    print("🔍 Đang phân đoạn tóc...")
    hair_mask = segmentator.segment_hair(image_path)
    
    # Lưu mask
    segmentator.save_mask(hair_mask, "hair_mask.png")
    
    # Tô màu tóc
    colors = {
        'đỏ': (255, 0, 0),
        'xanh_la': (0, 255, 0),
        'xanh_duong': (0, 0, 255),
        'vàng': (255, 255, 0),
        'tím': (255, 0, 255),
        'nâu': (165, 42, 42)
    }
    
    print("🎨 Đang tạo các màu tóc...")
    for color_name, color_value in colors.items():
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        output_path = f"hair_colored_{color_name}.jpg"
        cv2.imwrite(output_path, colored_image)
        print(f"✅ Đã tạo ảnh tóc màu {color_name}: {output_path}")
    
    print("\n🎉 Hoàn thành! Kiểm tra các file đầu ra.")

if __name__ == "__main__":
    main()
