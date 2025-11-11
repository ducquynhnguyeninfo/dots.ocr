#!/usr/bin/env python3
"""
Script to split PDF files into images, with each page saved as a separate image.
PDFs are organized in folders based on the filename pattern: ngu-van-kn-{grade}-{volume}.pdf
Images are saved in ./data/{grade}/{volume}/ directory structure.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Tuple, Optional

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Error: Required packages not found. Please install:")
    print("pip install pdf2image pillow")
    print("On Ubuntu/Debian, you may also need: sudo apt-get install poppler-utils")
    sys.exit(1)


def parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse PDF filename to extract grade and volume.
    Expected format: ngu-van-kn-{grade}-{volume}.pdf
    
    Args:
        filename: PDF filename
        
    Returns:
        Tuple of (grade, volume) or None if pattern doesn't match
    """
    # Remove .pdf extension
    base_name = filename.replace('.pdf', '')
    
    # Pattern: ngu-van-kn-{grade}-{volume}
    pattern = r'ngu-van-kn-(\d+)-(\d+)'
    match = re.match(pattern, base_name)
    
    if match:
        grade, volume = match.groups()
        return grade, volume
    
    return None


def split_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> bool:
    """
    Convert PDF pages to images and save them in the specified directory.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        dpi: DPI for image conversion
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Converting {pdf_path.name} to images...")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=dpi)
        
        # Save each page as an image
        for i, image in enumerate(images, 1):
            image_filename = f"page_{i}.jpg"
            image_path = output_dir / image_filename
            
            # Convert to RGB if necessary (some PDFs may have different color modes)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save with high quality
            image.save(str(image_path), 'JPEG', quality=95)
            print(f"  Saved: {image_path}")
        
        print(f"Successfully converted {len(images)} pages from {pdf_path.name}")
        return True
        
    except Exception as e:
        print(f"Error converting {pdf_path.name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Split PDF files into images organized by grade and volume"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data", 
        help="Directory containing PDF files (default: ./data)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=200, 
        help="DPI for image conversion (default: 200)"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true", 
        help="Show what would be done without actually converting files"
    )
    parser.add_argument(
        "--pdf_pattern", 
        type=str, 
        default="ngu-van-kn-*.pdf", 
        help="Pattern to match PDF files (default: ngu-van-kn-*.pdf)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        sys.exit(1)
    
    # Find PDF files matching the pattern
    pdf_files = list(data_dir.glob(args.pdf_pattern))
    
    if not pdf_files:
        print(f"No PDF files found matching pattern '{args.pdf_pattern}' in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    if args.dry_run:
        print("\nDry run - showing what would be done:")
        for pdf_file in pdf_files:
            grade_volume = parse_filename(pdf_file.name)
            if grade_volume:
                grade, volume = grade_volume
                output_dir = data_dir / grade / volume
                print(f"  {pdf_file.name} -> {output_dir}/page_*.jpg")
            else:
                print(f"  {pdf_file.name} -> SKIP (filename doesn't match expected pattern)")
        return
    
    # Process each PDF file
    success_count = 0
    for pdf_file in pdf_files:
        grade_volume = parse_filename(pdf_file.name)
        
        if not grade_volume:
            print(f"Skipping {pdf_file.name}: filename doesn't match expected pattern 'ngu-van-kn-{{grade}}-{{volume}}.pdf'")
            continue
        
        grade, volume = grade_volume
        output_dir = data_dir / grade / volume
        
        print(f"\nProcessing: {pdf_file.name}")
        print(f"Output directory: {output_dir}")
        
        if split_pdf_to_images(pdf_file, output_dir, args.dpi):
            success_count += 1
    
    print(f"\nCompleted! Successfully processed {success_count}/{len(pdf_files)} PDF files.")


if __name__ == "__main__":
    main()
