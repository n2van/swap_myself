import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class ContextPath(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(ContextPath, self).__init__()
        self.backbone_name = backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = nn.functional.interpolate(feat32_sum, size=feat16.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = nn.functional.interpolate(feat16_sum, size=feat8.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19, backbone='resnet18'):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath(backbone)
        self.ffm = FeatureFusionModule(256 + 128, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = nn.functional.interpolate(feat_out, size=(H, W), mode='bilinear', align_corners=True)
        feat_out16 = nn.functional.interpolate(feat_out16, size=(H, W), mode='bilinear', align_corners=True)
        feat_out32 = nn.functional.interpolate(feat_out32, size=(H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

def resnet18(pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
    
    features = []
    for name, module in model.named_children():
        if name == 'conv1':
            features.append(module)
        elif name == 'bn1':
            features.append(module)
        elif name == 'relu':
            features.append(module)
        elif name == 'maxpool':
            features.append(module)
        elif name == 'layer1':
            features.append(module)
        elif name == 'layer2':
            features.append(module)
        elif name == 'layer3':
            features.append(module)
        elif name == 'layer4':
            features.append(module)
    
    backbone = torch.nn.Sequential(*features)
    
    def forward(x):
        feat4 = backbone[:6](x)  # layer1
        feat8 = backbone[6](feat4)  # layer2
        feat16 = backbone[7](feat8)  # layer3
        feat32 = backbone[8](feat16)  # layer4
        return feat8, feat16, feat32
    
    backbone.forward = forward
    return backbone

def extract_hair_mask(image_path, model_path, output_path=None):
    """
    Extract hair mask from an image
    
    Args:
        image_path: path to the input image
        model_path: path to the pretrained model
        output_path: path to save the output mask (optional)
    
    Returns:
        hair_mask: binary mask of hair region
    """
    # Load model
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    
    # Use GPU if available
    if torch.cuda.is_available():
        net = net.cuda()
    
    # Load and preprocess image
    img = Image.open(image_path)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Resize to 512x512
    img = img.resize((512, 512), Image.BILINEAR)
    img_tensor = to_tensor(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Forward pass
    with torch.no_grad():
        out = net(img_tensor)[0]
    
    # Hair class index is 17 in CelebAMask-HQ
    hair_idx = 17
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
    # Create hair mask
    hair_mask = np.zeros_like(parsing)
    hair_mask[parsing == hair_idx] = 255
    
    # Save output if specified
    if output_path:
        cv2.imwrite(output_path, hair_mask)
    
    return hair_mask

def apply_hair_color(image_path, hair_mask_path, color=(0, 0, 255), output_path=None):
    """
    Apply color to hair region
    
    Args:
        image_path: path to the original image
        hair_mask_path: path to the hair mask
        color: RGB color tuple to apply to hair (default: red)
        output_path: path to save the colored image (optional)
    
    Returns:
        colored_img: image with colored hair
    """
    # Load image and mask
    img = cv2.imread(image_path)
    hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize mask to match image if needed
    if img.shape[:2] != hair_mask.shape[:2]:
        hair_mask = cv2.resize(hair_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create color overlay
    color_overlay = np.zeros_like(img)
    color_overlay[:] = color[::-1]  # BGR format for OpenCV
    
    # Apply color to hair region
    mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
    colored_img = img * (1 - mask_3d * 0.7) + color_overlay * mask_3d * 0.7
    colored_img = colored_img.astype(np.uint8)
    
    # Save output if specified
    if output_path:
        cv2.imwrite(output_path, colored_img)
    
    return colored_img

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--output', type=str, default='hair_mask.png', help='Path to output hair mask')
    parser.add_argument('--color', type=str, default=None, help='Color to apply to hair (R,G,B format)')
    parser.add_argument('--colored_output', type=str, default=None, help='Path to save colored hair image')
    
    args = parser.parse_args()
    
    # Extract hair mask
    hair_mask = extract_hair_mask(args.input, args.model, args.output)
    
    # Apply color if specified
    if args.color and args.colored_output:
        color = tuple(map(int, args.color.split(',')))
        apply_hair_color(args.input, args.output, color, args.colored_output)
        print(f"Colored hair image saved to {args.colored_output}")
    
    print(f"Hair mask saved to {args.output}") 