
# 🎬 Complete Video Watermark Remover & Upscaler

A powerful Python-based tool to automatically remove watermarks from videos and upscale them using AI models. This tool provides a complete pipeline: extract frames → remove watermarks → upscale → create clean video.

## ✨ Features

- **🎯 Watermark Removal**: Remove watermarks using AI (LaMa) or OpenCV inpainting
- **🚀 AI Upscaling**: Upscale videos to HD (2x) or 4K (4x) using RealESRGAN
- **🎨 Quality Enhancement**: CLAHE contrast enhancement and sharpening
- **📹 Complete Pipeline**: Automated video processing from input to output
- **🎛️ Flexible Processing**: Process entire videos or individual frames
- **🖼️ Single Image Support**: Process individual images as well

## 🛠️ Technologies Used

- **LaMa (Large Mask Inpainting)**: State-of-the-art AI for watermark removal
- **RealESRGAN**: Advanced AI for image upscaling
- **OpenCV**: Computer vision for fallback inpainting
- **FFmpeg**: Video frame extraction and encoding
- **PIL/Pillow**: Image processing

## 📋 Requirements

- Python 3.12+
- FFmpeg (pre-installed in Replit environment)
- CUDA-capable GPU (optional, for faster processing)

## 🚀 Installation

### On Replit (Recommended)

1. Fork this repository to your Replit account
2. Click the **Run** button - dependencies will auto-install

### Local Installation

```bash
# Clone the repository
git clone https://github.com/hokireceh/watermark-remover.git
cd watermark-remover

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

```
opencv-python-headless==4.8.1.78
numpy==1.24.3
Pillow==10.0.0
simple-lama-inpainting==0.1.1
realesrgan==0.3.0
basicsr==1.4.2
torch==2.0.1
torchvision==0.15.2
```

## 🎯 Usage

### Interactive Menu

Run the main script:

```bash
python main.py
```

You'll see an interactive menu with the following options:

```
📋 Main Menu:
  1. Process Video (Auto: Extract → Process → Create)
