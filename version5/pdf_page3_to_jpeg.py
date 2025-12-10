#!/usr/bin/env python3
"""
Extract page 3 from all PDFs in a folder and convert to high-quality JPEGs
"""

import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image


def extract_page3_to_jpeg(pdf_path, output_folder, dpi=300, quality=95):
    """
    Extract page 3 from a PDF and save as high-quality JPEG
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save the JPEG
        dpi: Resolution (default 300 for high quality)
        quality: JPEG quality 1-100 (default 95 for high quality)
    
    Returns:
        Path to saved JPEG or None if failed
    """
    try:
        pdf_name = Path(pdf_path).stem
        
        # Convert only page 3 (first_page=3, last_page=3)
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=3,
            last_page=3,
            fmt='jpeg'
        )
        
        if not images:
            print(f"  ❌ {pdf_name}: Page 3 not found")
            return None
        
        # Save the image
        output_path = Path(output_folder) / f"{pdf_name}_page3.jpg"
        images[0].save(
            output_path,
            'JPEG',
            quality=quality,
            optimize=True
        )
        
        print(f"  ✓ {pdf_name}: Saved to {output_path.name}")
        return output_path
        
    except Exception as e:
        print(f"  ❌ {Path(pdf_path).name}: {str(e)}")
        return None


def batch_process_pdfs(input_folder, output_folder=None, dpi=300, quality=95):
    """
    Process all PDFs in a folder
    
    Args:
        input_folder: Folder containing PDFs
        output_folder: Folder to save JPEGs (default: input_folder/output)
        dpi: Resolution for conversion
        quality: JPEG quality
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    # Set default output folder
    if output_folder is None:
        output_path = input_path / "page3_jpegs"
    else:
        output_path = Path(output_folder)
    
    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = sorted(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s)")
    print(f"Output folder: {output_path}")
    print(f"Settings: DPI={dpi}, Quality={quality}")
    print("\nProcessing...\n")
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        result = extract_page3_to_jpeg(pdf_file, output_path, dpi, quality)
        if result:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  ✓ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Output folder: {output_path.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "C:/MEHBOOB HD/Roshn Backup/Demarcation_Verifier/version5/input_data"  # Change this to your folder
    OUTPUT_FOLDER = None  # None = creates 'page3_jpegs' subfolder in input folder
    DPI = 300  # 300 DPI for high quality (use 150 for faster/smaller files)
    QUALITY = 95  # JPEG quality 1-100 (95 is high quality, 85 is good balance)
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        INPUT_FOLDER = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FOLDER = sys.argv[2]
    if len(sys.argv) > 3:
        DPI = int(sys.argv[3])
    if len(sys.argv) > 4:
        QUALITY = int(sys.argv[4])
    
    batch_process_pdfs(INPUT_FOLDER, OUTPUT_FOLDER, DPI, QUALITY)