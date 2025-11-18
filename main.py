
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
import sys
from typing import Optional, Tuple

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

# Constants
WATERMARK_X_RATIO = 0.28
WATERMARK_Y_RATIO = 0.840
WATERMARK_W_RATIO = 0.47
WATERMARK_H_RATIO = 0.045
DEFAULT_FPS = 30.0
DEFAULT_QUALITY = "high"
QUALITY_CRF = {"high": 18, "medium": 23, "low": 28}

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
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize AI models with error handling"""
        if LAMA_AVAILABLE:
            try:
                print("🔄 Loading LaMa AI...")
                self.lama = SimpleLama()
                print("✓ LaMa ready!")
            except Exception as e:
                print(f"⚠️  Failed to load LaMa: {e}")
                self.lama = None
        
        if REALESRGAN_AVAILABLE:
            try:
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
            except Exception as e:
                print(f"⚠️  Failed to load RealESRGAN: {e}")
                self.upscaler = None
    
    def _validate_video_path(self, video_path: str) -> bool:
        """Validate if video file exists and is accessible"""
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return False
        
        if not os.path.isfile(video_path):
            print(f"❌ Error: Path is not a file: {video_path}")
            return False
        
        # Check if file is a video by extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"⚠️  Warning: File may not be a video file: {video_path}")
        
        return True
    
    def _validate_image_path(self, image_path: str) -> bool:
        """Validate if image file exists and is accessible"""
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return False
        
        if not os.path.isfile(image_path):
            print(f"❌ Error: Path is not a file: {image_path}")
            return False
        
        return True
    
    def extract_frames(self, video_path: str, fps: Optional[float] = None) -> Tuple[bool, float]:
        """Extract frames from video"""
        if not self._validate_video_path(video_path):
            return False, 0.0
        
        print(f"\n📹 Extracting frames from: {video_path}")
        
        try:
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error: Cannot open video file: {video_path}")
                return False, 0.0
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if original_fps <= 0:
                print(f"❌ Error: Invalid FPS detected: {original_fps}")
                return False, 0.0
            
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
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'fps={fps}',
                f'{self.frames_folder}/frame_%04d.png'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Error extracting frames: {result.stderr}")
                return False, original_fps
            
            frame_count = len(list(Path(self.frames_folder).glob("*.png")))
            if frame_count == 0:
                print("❌ Error: No frames were extracted")
                return False, original_fps
            
            print(f"✓ Extracted {frame_count} frames")
            return True, original_fps
        
        except Exception as e:
            print(f"❌ Unexpected error during frame extraction: {e}")
            return False, 0.0
    
    def remove_watermark_lama(self, image_path: str) -> Optional[Image.Image]:
        """Remove watermark using LaMa AI"""
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            
            # Watermark position
            x = int(width * WATERMARK_X_RATIO)
            y = int(height * WATERMARK_Y_RATIO)
            w = int(width * WATERMARK_W_RATIO)
            h = int(height * WATERMARK_H_RATIO)
            
            # Create mask
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(x, y), (x+w, y+h)], fill=255)
            
            # Remove watermark
            result = self.lama(img, mask)
            return result
        
        except Exception as e:
            print(f"⚠️  LaMa processing error: {e}")
            return None
    
    def remove_watermark_opencv(self, image_path: str) -> Optional[Image.Image]:
        """Fallback: OpenCV inpainting"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Error: Cannot read image: {image_path}")
                return None
            
            height, width = img.shape[:2]
            
            x = int(width * WATERMARK_X_RATIO)
            y = int(height * WATERMARK_Y_RATIO)
            w = int(width * WATERMARK_W_RATIO)
            h = int(height * WATERMARK_H_RATIO)
            
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            
            result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        except Exception as e:
            print(f"❌ OpenCV processing error: {e}")
            return None
    
    def upscale_image(self, img: Image.Image, scale: int = 2) -> Image.Image:
        """Upscale with RealESRGAN or LANCZOS"""
        try:
            if REALESRGAN_AVAILABLE and self.upscaler and scale == 4:
                img_np = np.array(img)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                output, _ = self.upscaler.enhance(img_cv, outscale=scale)
                return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            else:
                width, height = img.size
                return img.resize((width * scale, height * scale), Image.LANCZOS)
        
        except Exception as e:
            print(f"⚠️  Upscaling error (fallback to original): {e}")
            return img
    
    def enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance image quality"""
        try:
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
        
        except Exception as e:
            print(f"⚠️  Enhancement error (fallback to original): {e}")
            return img
    
    def process_frames(self, upscale: bool = True, enhance: bool = True, scale: int = 2) -> bool:
        """Process all frames"""
        frames = sorted(Path(self.frames_folder).glob("*.png"))
        
        if not frames:
            print("❌ No frames found!")
            return False
        
        print(f"\n🎨 Processing {len(frames)} frames...")
        
        # Clear processed folder
        shutil.rmtree(self.processed_folder, ignore_errors=True)
        Path(self.processed_folder).mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        for i, frame_path in enumerate(frames, 1):
            try:
                print(f"[{i}/{len(frames)}] {frame_path.name}", end=" ")
                
                # Remove watermark
                if LAMA_AVAILABLE and self.lama:
                    img = self.remove_watermark_lama(str(frame_path))
                else:
                    img = self.remove_watermark_opencv(str(frame_path))
                
                if img is None:
                    print("❌ Failed")
                    failed_count += 1
                    continue
                
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
                processed_count += 1
            
            except Exception as e:
                print(f"❌ Error: {e}")
                failed_count += 1
                continue
        
        print(f"\n✓ Processed {processed_count} frames successfully")
        if failed_count > 0:
            print(f"⚠️  {failed_count} frames failed to process")
        
        return processed_count > 0
    
    def create_video(self, output_name: str = "output_clean.mp4", fps: float = DEFAULT_FPS, quality: str = DEFAULT_QUALITY) -> bool:
        """Combine frames into video"""
        print(f"\n🎬 Creating video: {output_name}")
        
        frames = sorted(Path(self.processed_folder).glob("*.png"))
        if not frames:
            print("❌ No processed frames found!")
            return False
        
        output_path = f"{self.output_folder}/{output_name}"
        
        # Quality presets
        crf = QUALITY_CRF.get(quality, 18)
        
        try:
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
        
        except Exception as e:
            print(f"❌ Unexpected error creating video: {e}")
            return False
    
    def process_video(self, video_path: str, upscale: bool = True, enhance: bool = True, 
                     scale: int = 2, fps: Optional[float] = None, quality: str = DEFAULT_QUALITY) -> bool:
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


def print_menu() -> None:
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


def get_validated_input(prompt: str, default: str = "", valid_options: Optional[list] = None) -> str:
    """Get and validate user input"""
    while True:
        user_input = input(prompt).strip() or default
        
        if valid_options is None or user_input in valid_options:
            return user_input
        
        print(f"❌ Invalid input. Please choose from: {', '.join(valid_options)}")


def main() -> None:
    """Main function with interactive menu"""
    try:
        processor = VideoWatermarkRemover()
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    while True:
        try:
            print_menu()
            choice = input("Choose option: ").strip()
            
            if choice == "1":
                # Complete pipeline
                video_path = input("\n📹 Video path: ").strip()
                if not video_path:
                    print("❌ Video path cannot be empty")
                    continue
                
                print("\nOptions:")
                print("  Upscale: 1=No, 2=2x (HD), 4=4x (4K)")
                scale_choice = get_validated_input("  Choose (default=2): ", "2", ["1", "2", "4"])
                scale = int(scale_choice)
                upscale = scale > 1
                
                enhance = get_validated_input("  Enhance quality? (y/n, default=y): ", "y", ["y", "n"]) != 'n'
                
                fps_input = input("  FPS (default=original): ").strip()
                fps = float(fps_input) if fps_input else None
                
                print("  Quality: high/medium/low")
                quality = get_validated_input("  Choose (default=high): ", "high", ["high", "medium", "low"])
                
                processor.process_video(video_path, upscale, enhance, scale, fps, quality)
            
            elif choice == "2":
                # Extract only
                video_path = input("\n📹 Video path: ").strip()
                if not video_path:
                    print("❌ Video path cannot be empty")
                    continue
                
                fps_input = input("  FPS (default=original): ").strip()
                fps = float(fps_input) if fps_input else None
                processor.extract_frames(video_path, fps)
            
            elif choice == "3":
                # Process frames only
                scale_choice = get_validated_input("\n  Scale (1/2/4, default=2): ", "2", ["1", "2", "4"])
                scale = int(scale_choice)
                upscale = scale > 1
                enhance = get_validated_input("  Enhance? (y/n, default=y): ", "y", ["y", "n"]) != 'n'
                processor.process_frames(upscale, enhance, scale)
            
            elif choice == "4":
                # Create video only
                name = input("\n  Output name (default=output_clean.mp4): ").strip() or "output_clean.mp4"
                fps_str = input("  FPS (default=30): ").strip() or "30"
                fps = float(fps_str)
                quality = get_validated_input("  Quality (high/medium/low, default=high): ", "high", ["high", "medium", "low"])
                processor.create_video(name, fps, quality)
            
            elif choice == "5":
                # Single image
                img_path = input("\n  Image path: ").strip()
                if not img_path:
                    print("❌ Image path cannot be empty")
                    continue
                
                if not processor._validate_image_path(img_path):
                    continue
                
                output_path = input("  Output path (default=output.png): ").strip() or "output.png"
                
                print("  Processing...")
                if LAMA_AVAILABLE and processor.lama:
                    img = processor.remove_watermark_lama(img_path)
                else:
                    img = processor.remove_watermark_opencv(img_path)
                
                if img is None:
                    print("❌ Failed to process image")
                    continue
                
                upscale_choice = get_validated_input("  Upscale? (1/2/4, default=1): ", "1", ["1", "2", "4"])
                if int(upscale_choice) > 1:
                    img = processor.upscale_image(img, int(upscale_choice))
                
                img.save(output_path, quality=95)
                print(f"  ✓ Saved: {output_path}")
            
            elif choice == "0":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option!")
            
            input("\nPress Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
