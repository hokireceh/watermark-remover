#!/usr/bin/env python3
"""
Complete Video Watermark Remover & Upscaler
Extract video → Remove watermark → Upscale → Create clean video
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import subprocess
from pathlib import Path
import shutil

try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

class VideoWatermarkRemover:
    def __init__(self):
        self.base_folder = "video_project"
        self.frames_folder = f"{self.base_folder}/frames"
        self.processed_folder = f"{self.base_folder}/processed"
        self.output_folder = f"{self.base_folder}/output"
        
        # Create folders
        for folder in [self.frames_folder, self.processed_folder, self.output_folder]:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize AI models
        self.lama = None
        self.upscaler = None
        
        if LAMA_AVAILABLE:
            print("🔄 Loading LaMa AI...")
            self.lama = SimpleLama()
            print("✓ LaMa ready!")
        
        if REALESRGAN_AVAILABLE:
            print("🔄 Loading RealESRGAN...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            print("✓ RealESRGAN ready!")
    
    def extract_frames(self, video_path, fps=None):
        """Extract frames from video"""
        print(f"\n📹 Extracting frames from: {video_path}")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps is None:
            fps = original_fps
        
        print(f"   Video FPS: {original_fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Extracting at: {fps} fps")
        
        # Clear old frames
        shutil.rmtree(self.frames_folder, ignore_errors=True)
        Path(self.frames_folder).mkdir(parents=True, exist_ok=True)
        
        # Extract with ffmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps}',
            f'{self.frames_folder}/frame_%04d.png'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error extracting frames: {result.stderr}")
            return False, original_fps
        
        frame_count = len(list(Path(self.frames_folder).glob("*.png")))
        print(f"✓ Extracted {frame_count} frames")
        
        return True, original_fps
    
    def remove_watermark_lama(self, image_path):
        """Remove watermark using LaMa AI"""
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        # Watermark position
        x = int(width * 0.28)
        y = int(height * 0.840)
        w = int(width * 0.47)
        h = int(height * 0.045)
        
        # Create mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x+w, y+h], fill=255)
        
        # Remove watermark
        result = self.lama(img, mask)
        return result
    
    def remove_watermark_opencv(self, image_path):
        """Fallback: OpenCV inpainting"""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        x = int(width * 0.28)
        y = int(height * 0.840)
        w = int(width * 0.47)
        h = int(height * 0.045)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    def upscale_image(self, img, scale=2):
        """Upscale with RealESRGAN or LANCZOS"""
        if REALESRGAN_AVAILABLE and self.upscaler and scale == 4:
            img_np = np.array(img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            output, _ = self.upscaler.enhance(img_cv, outscale=scale)
            return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        else:
            width, height = img.size
            return img.resize((width * scale, height * scale), Image.LANCZOS)
    
    def enhance_image(self, img):
        """Enhance image quality"""
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    def process_frames(self, upscale=True, enhance=True, scale=2):
        """Process all frames"""
        frames = sorted(Path(self.frames_folder).glob("*.png"))
        
        if not frames:
            print("❌ No frames found!")
            return False
        
        print(f"\n🎨 Processing {len(frames)} frames...")
        
        # Clear processed folder
        shutil.rmtree(self.processed_folder, ignore_errors=True)
        Path(self.processed_folder).mkdir(parents=True, exist_ok=True)
        
        for i, frame_path in enumerate(frames, 1):
            print(f"[{i}/{len(frames)}] {frame_path.name}", end=" ")
            
            # Remove watermark
            if LAMA_AVAILABLE and self.lama:
                img = self.remove_watermark_lama(str(frame_path))
            else:
                img = self.remove_watermark_opencv(str(frame_path))
            
            # Upscale
            if upscale:
                print("→ Upscaling", end=" ")
                img = self.upscale_image(img, scale)
            
            # Enhance
            if enhance:
                print("→ Enhancing", end=" ")
                img = self.enhance_image(img)
            
            # Save
            output_path = Path(self.processed_folder) / frame_path.name
            img.save(output_path, quality=95)
            print("✓")
        
        print(f"\n✓ All frames processed!")
        return True
    
    def create_video(self, output_name="output_clean.mp4", fps=30, quality="high"):
        """Combine frames into video"""
        print(f"\n🎬 Creating video: {output_name}")
        
        frames = sorted(Path(self.processed_folder).glob("*.png"))
        if not frames:
            print("❌ No processed frames found!")
            return False
        
        output_path = f"{self.output_folder}/{output_name}"
        
        # Quality presets
        crf = {"high": 18, "medium": 23, "low": 28}.get(quality, 18)
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', f'{self.processed_folder}/*.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', str(crf),
            '-preset', 'slow',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error creating video: {result.stderr}")
            return False
        
        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Video created: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return True
    
    def process_video(self, video_path, upscale=True, enhance=True, scale=2, fps=None, quality="high"):
        """Complete pipeline: video in → clean video out"""
        print("\n" + "="*60)
        print("        Complete Video Processing Pipeline")
        print("="*60)
        
        # Step 1: Extract frames
        success, original_fps = self.extract_frames(video_path, fps)
        if not success:
            return False
        
        # Step 2: Process frames
        success = self.process_frames(upscale, enhance, scale)
        if not success:
            return False
        
        # Step 3: Create video
        video_name = Path(video_path).stem
        output_name = f"{video_name}_clean_{'4K' if scale==4 else 'HD'}.mp4"
        fps_output = fps if fps else original_fps
        
        success = self.create_video(output_name, fps_output, quality)
        
        if success:
            print("\n" + "="*60)
            print("✓ COMPLETE! Video processed successfully!")
            print(f"  Output: {self.output_folder}/{output_name}")
            print("="*60)
        
        return success


def print_menu():
    """Print main menu"""
    print("\n" + "="*60)
    print("     Complete Video Watermark Remover & Upscaler")
    print("="*60)
    print("\n📋 Main Menu:")
    print("  1. Process Video (Auto: Extract → Process → Create)")
    print("  2. Extract Frames Only")
    print("  3. Process Frames Only")
    print("  4. Create Video from Processed Frames")
    print("  5. Process Single Image")
    print("  0. Exit")
    print("\n" + "-"*60)
    
    if not LAMA_AVAILABLE:
        print("⚠️  LaMa not installed (pip install simple-lama-inpainting)")
    if not REALESRGAN_AVAILABLE:
        print("⚠️  RealESRGAN not installed (pip install realesrgan)")
    print()


def main():
    """Main function with interactive menu"""
    processor = VideoWatermarkRemover()
    
    while True:
        print_menu()
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            # Complete pipeline
            video_path = input("\n📹 Video path: ").strip()
            
            print("\nOptions:")
            print("  Upscale: 1=No, 2=2x (HD), 4=4x (4K)")
            scale_choice = input("  Choose (default=2): ").strip() or "2"
            scale = int(scale_choice) if scale_choice in ["1","2","4"] else 2
            upscale = scale > 1
            
            enhance = input("  Enhance quality? (y/n, default=y): ").strip().lower() != 'n'
            
            fps_input = input("  FPS (default=original): ").strip()
            fps = float(fps_input) if fps_input else None
            
            print("  Quality: high/medium/low")
            quality = input("  Choose (default=high): ").strip() or "high"
            
            processor.process_video(video_path, upscale, enhance, scale, fps, quality)
        
        elif choice == "2":
            # Extract only
            video_path = input("\n📹 Video path: ").strip()
            fps_input = input("  FPS (default=original): ").strip()
            fps = float(fps_input) if fps_input else None
            processor.extract_frames(video_path, fps)
        
        elif choice == "3":
            # Process frames only
            scale_choice = input("\n  Scale (1/2/4, default=2): ").strip() or "2"
            scale = int(scale_choice)
            upscale = scale > 1
            enhance = input("  Enhance? (y/n, default=y): ").strip().lower() != 'n'
            processor.process_frames(upscale, enhance, scale)
        
        elif choice == "4":
            # Create video only
            name = input("\n  Output name (default=output_clean.mp4): ").strip() or "output_clean.mp4"
            fps = float(input("  FPS (default=30): ").strip() or "30")
            quality = input("  Quality (high/medium/low, default=high): ").strip() or "high"
            processor.create_video(name, fps, quality)
        
        elif choice == "5":
            # Single image
            img_path = input("\n  Image path: ").strip()
            output_path = input("  Output path (default=output.png): ").strip() or "output.png"
            
            print("  Processing...")
            if LAMA_AVAILABLE and processor.lama:
                img = processor.remove_watermark_lama(img_path)
            else:
                img = processor.remove_watermark_opencv(img_path)
            
            upscale_choice = input("  Upscale? (1/2/4, default=1): ").strip() or "1"
            if int(upscale_choice) > 1:
                img = processor.upscale_image(img, int(upscale_choice))
            
            img.save(output_path, quality=95)
            print(f"  ✓ Saved: {output_path}")
        
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option!")
        
        input("\n Press Enter to continue...")


if __name__ == "__main__":
    main()
