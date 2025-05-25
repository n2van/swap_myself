# Hair Segmentation

This project provides tools to segment hair from facial images and apply different hair colors. It's based on the [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) repository.

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy opencv-python pillow
```

## Download Pre-trained Model

Download the pre-trained model from the original repository:

1. Create a `pretrained` directory:
   ```bash
   mkdir pretrained
   ```

2. Download the pre-trained model:
   ```bash
   # You can download it manually from the link below
   # https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
   ```

   Save the downloaded file as `pretrained/79999_iter.pth`

## Usage

### Extract Hair Mask

```bash
python hair_segmentation.py --input path/to/image.jpg --model pretrained/79999_iter.pth --output hair_mask.png
```

### Apply Hair Color

```bash
python hair_segmentation.py --input path/to/image.jpg --model pretrained/79999_iter.pth --output hair_mask.png --color 255,0,0 --colored_output colored_hair.png
```

The `--color` parameter accepts RGB values in the format `R,G,B`.

## Example Colors

- Red: 255,0,0
- Green: 0,255,0
- Blue: 0,0,255
- Yellow: 255,255,0
- Purple: 255,0,255
- Cyan: 0,255,255
- Black: 0,0,0
- White: 255,255,255
- Brown: 165,42,42
- Pink: 255,192,203
- Orange: 255,165,0

## Credits

This project is based on the [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) repository. 