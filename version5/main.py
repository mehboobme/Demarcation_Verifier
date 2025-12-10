#!/usr/bin/env python3
"""
PDF Validation System - Main Entry Point
=========================================

This system validates ROSHN property PDFs against ground truth data in Excel.

Features:
- File name verification (Page 1)
- Header/footer consistency across all pages
- Red highlight detection for neighborhood and plot maps (Page 2)
- Area table extraction and validation (Page 3)
- Dimension extraction (Page 3)
- Floor area verification (Pages 4-6)
- Facade style detection (Modern vs Traditional) (Pages 8-9)
- Comprehensive reporting (Excel, JSON, Text)

Usage:
    python main.py --pdf <pdf_path> --excel <excel_path> [--output <output_dir>]
    python main.py --pdf-dir <pdf_directory> --excel <excel_path> [--output <output_dir>]
    
For batch processing:
    python main.py --pdf-dir /path/to/pdfs --excel /path/to/ground_truth.xlsx
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_extractor import PDFExtractor, ExtractedData, extract_pdf_data
from excel_loader import GroundTruthLoader, GroundTruthRecord
from validator import Validator, ValidationReport, validate_pdf_against_ground_truth
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validation.log')
    ]
)
logger = logging.getLogger(__name__)


class PDFValidationSystem:
    """Main validation system orchestrator"""
    
    def __init__(self, excel_path: str, output_dir: str = None, debug: bool = True, use_gpu: bool = False):
        self.excel_path = excel_path
        self.output_dir = output_dir or "./output"
        self.debug_dir = "./debug_images"
        self.debug = debug
        self.use_gpu = use_gpu
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load ground truth
        logger.info(f"Loading ground truth from: {excel_path}")
        self.gt_loader = GroundTruthLoader(excel_path)
        self.ground_truth = self.gt_loader.load()
        logger.info(f"Loaded {len(self.ground_truth)} ground truth records")
        
        # Initialize components
        self.validator = Validator(use_gpu=self.use_gpu)
        self.report_generator = ReportGenerator(self.output_dir)
        
        # Store results
        self.results: List[ValidationReport] = []
        self.extraction_data: Dict[str, ExtractedData] = {}
        
    def validate_single_pdf(self, pdf_path: str) -> Optional[ValidationReport]:
        """Validate a single PDF against ground truth"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating: {pdf_path}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Extract data from PDF
            logger.info("Step 1: Extracting data from PDF...")
            pdf_debug_dir = os.path.join(self.debug_dir, Path(pdf_path).stem)
            extractor = PDFExtractor(pdf_path, pdf_debug_dir)
            pdf_data = extractor.extract_all()
            
            # Store extraction data
            self.extraction_data[pdf_path] = pdf_data
            
            # Log extracted data
            self._log_extracted_data(pdf_data)
            
            # Step 2: Find matching ground truth
            logger.info("Step 2: Finding matching ground truth...")
            gt_record = self._find_ground_truth(pdf_data)
            
            if not gt_record:
                logger.error(f"No ground truth found for plot {pdf_data.plot_number}")
                return None
            
            self._log_ground_truth(gt_record)
            
            # Step 3: Validate
            logger.info("Step 3: Running validation...")
            report = self.validator.validate(pdf_data, gt_record)
            
            # Log results
            self._log_validation_result(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating {pdf_path}: {e}", exc_info=True)
            return None
    
    def validate_directory(self, pdf_dir: str) -> List[ValidationReport]:
        """Validate all PDFs in a directory"""
        logger.info(f"Scanning directory: {pdf_dir}")
        
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        results = []
        for pdf_path in pdf_files:
            report = self.validate_single_pdf(str(pdf_path))
            if report:
                results.append(report)
        
        self.results = results
        return results
    
    def generate_reports(self, filename_prefix: str = "validation") -> Dict[str, str]:
        """Generate all report formats"""
        logger.info("Generating reports...")
        paths = self.report_generator.generate_all_reports(self.results, filename_prefix)
        
        logger.info(f"Report generated: {paths['excel']}")
        
        return paths
    
    def _find_ground_truth(self, pdf_data: ExtractedData) -> Optional[GroundTruthRecord]:
        """Find matching ground truth record"""
        # Try by plot number first
        if pdf_data.plot_number:
            record = self.ground_truth.get(pdf_data.plot_number)
            if record:
                logger.info(f"Found ground truth by plot number: {pdf_data.plot_number}")
                return record
        
        # Try by unit code
        if pdf_data.unit_code:
            for record in self.ground_truth.values():
                if record.unit_code == pdf_data.unit_code:
                    logger.info(f"Found ground truth by unit code: {pdf_data.unit_code}")
                    return record
        
        return None
    
    def _log_extracted_data(self, data: ExtractedData):
        """Log extracted PDF data"""
        logger.info("\nExtracted PDF Data:")
        logger.info(f"  Filename: {data.filename}")
        logger.info(f"  Unit Code: {data.unit_code}")
        logger.info(f"  Plot Number: {data.plot_number}")
        logger.info(f"  Unit Type: {data.unit_type}")
        logger.info(f"  Land Area: {data.land_area} m²")
        logger.info(f"  Ground Floor: {data.ground_floor_area} m²")
        logger.info(f"  First Floor: {data.first_floor_area} m²")
        logger.info(f"  Upper Floor: {data.upper_floor_area} m²")
        logger.info(f"  Total GFA: {data.total_floor_area} m²")
        logger.info(f"  Dimensions: {data.land_length} x {data.land_width} m")
        logger.info(f"  Street Width: {data.street_width} m")
        logger.info(f"  Facade Type: {data.facade_type}")
        logger.info(f"  Red Highlight Neighborhood: {data.red_highlight_neighborhood}")
        logger.info(f"  Red Highlight Plot: {data.red_highlight_plot}")
        
        if data.extraction_notes:
            logger.warning(f"  Extraction Notes: {data.extraction_notes}")
    
    def _log_ground_truth(self, record: GroundTruthRecord):
        """Log ground truth data"""
        logger.info("\nGround Truth Data:")
        logger.info(f"  Unit Code: {record.unit_code}")
        logger.info(f"  Plot Number: {record.amana_plot_no}")
        logger.info(f"  Unit Type: {record.unit_type}")
        logger.info(f"  Land Area: {record.land_area} m²")
        logger.info(f"  Ground Floor: {record.ground_floor_area} m²")
        logger.info(f"  First Floor: {record.first_floor_area} m²")
        logger.info(f"  Second Floor: {record.second_floor_area} m²")
        logger.info(f"  GFA: {record.gfa} m²")
        logger.info(f"  Dimensions: {record.land_length} x {record.land_width} m")
        logger.info(f"  Street Width: {record.street_width} m")
        logger.info(f"  Facade Type: {record.facade_type}")
    
    def _log_validation_result(self, report: ValidationReport):
        """Log validation results"""
        logger.info("\nValidation Results:")
        logger.info(f"  Overall Status: {report.overall_status}")
        logger.info(f"  Total Checks: {report.total_checks}")
        logger.info(f"  Matches: {report.matches}")
        logger.info(f"  Mismatches: {report.mismatches}")
        logger.info(f"  Warnings: {report.warnings}")
        
        if report.critical_failures:
            logger.error(f"  CRITICAL FAILURES ({len(report.critical_failures)}):")
            for f in report.critical_failures:
                logger.error(f"    - {f.field_name}: {f.message}")
        
        if report.high_failures:
            logger.warning(f"  HIGH FAILURES ({len(report.high_failures)}):")
            for f in report.high_failures:
                logger.warning(f"    - {f.field_name}: {f.message}")
        
        if report.medium_failures:
            logger.info(f"  MEDIUM FAILURES ({len(report.medium_failures)}):")
            for f in report.medium_failures:
                logger.info(f"    - {f.field_name}: {f.message}")


# Default paths
DEFAULT_PDF_DIR = "./input_data"
DEFAULT_EXCEL_PATH = "./input_data/ground_truth.xlsx"
DEFAULT_OUTPUT_DIR = "./output"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PDF Validation System - Validates ROSHN property PDFs against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive, but not required - defaults used)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--pdf", type=str, help="Path to single PDF file to validate")
    input_group.add_argument("--pdf-dir", type=str, default=DEFAULT_PDF_DIR, help="Path to directory containing PDFs to validate")
    
    parser.add_argument("--excel", type=str, default=DEFAULT_EXCEL_PATH, help="Path to ground truth Excel file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--gpu", action="store_true", help="Use GPU-accelerated verifier with blue boundary cropping")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validation system
    system = PDFValidationSystem(
        excel_path=args.excel,
        output_dir=args.output,
        debug=args.debug,
        use_gpu=args.gpu
    )
    
    # Run validation
    if args.pdf:
        report = system.validate_single_pdf(args.pdf)
        if report:
            system.results = [report]
    else:
        system.validate_directory(args.pdf_dir)
    
    # Generate reports
    if system.results:
        paths = system.generate_reports()
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"PDFs Validated: {len(system.results)}")
        
        total_matches = sum(r.matches for r in system.results)
        total_mismatches = sum(r.mismatches for r in system.results)
        total_checks = sum(r.total_checks for r in system.results)
        
        print(f"Total Checks: {total_checks}")
        print(f"Total Matches: {total_matches}")
        print(f"Total Mismatches: {total_mismatches}")
        print(f"Match Rate: {100*total_matches/total_checks:.1f}%" if total_checks else "N/A")
        
        print(f"\nReport saved to: {paths['excel']}")
    else:
        print("No validation results generated")


if __name__ == "__main__":
    main()
