"""
Validator Module
Compares extracted PDF data against ground truth and generates validation results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from pdf_extractor import ExtractedData
from excel_loader import GroundTruthRecord
from config import VALIDATION_RULES, SEVERITY, PDF_TO_EXCEL_MAPPING

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    WARNING = "WARNING"
    NOT_FOUND = "NOT_FOUND"
    SKIPPED = "SKIPPED"


@dataclass
class ValidationResult:
    """Result of a single field validation"""
    field_name: str
    pdf_value: Any
    ground_truth_value: Any
    status: ValidationStatus
    severity: int
    message: str = ""
    page_number: int = 0


@dataclass
class ValidationReport:
    """Complete validation report for a PDF"""
    pdf_filename: str
    unit_code: str
    plot_number: int
    
    total_checks: int = 0
    matches: int = 0
    mismatches: int = 0
    warnings: int = 0
    not_found: int = 0
    
    critical_failures: List[ValidationResult] = field(default_factory=list)
    high_failures: List[ValidationResult] = field(default_factory=list)
    medium_failures: List[ValidationResult] = field(default_factory=list)
    low_failures: List[ValidationResult] = field(default_factory=list)
    
    all_results: List[ValidationResult] = field(default_factory=list)
    
    overall_status: str = "UNKNOWN"
    summary: str = ""


class Validator:
    """Validates PDF data against ground truth"""
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def validate(self, pdf_data: ExtractedData, ground_truth: GroundTruthRecord) -> ValidationReport:
        """Main validation method"""
        logger.info(f"Validating: {pdf_data.unit_code}")
        
        report = ValidationReport(
            pdf_filename=pdf_data.filename,
            unit_code=pdf_data.unit_code,
            plot_number=pdf_data.plot_number
        )
        
        # Define validation checks
        checks = [
            # ============ PAGE 1: Unit Code ============
            ("page1_unit_code", pdf_data.page1_unit_code, ground_truth.unit_code, 1),
            
            # ============ PAGE 2 Section 1: NBHD_Name, AMANA_Plot_No ============
            ("page2_sec1_nbhd_name", pdf_data.page2_sec1_nbhd_name, ground_truth.nbhd_name, 2),
            
            # ============ PAGE 2 Section 2: NBHD_Name, Unit_Type, AMANA_Plot_No (RED plot) ============
            ("page2_sec2_nbhd_name", pdf_data.page2_sec2_nbhd_name, ground_truth.nbhd_name, 2),
            ("page2_sec2_unit_type", pdf_data.page2_sec2_unit_type, ground_truth.unit_type, 1),
            ("page2_sec2_amana_plot_no", pdf_data.page2_sec2_amana_plot_no, ground_truth.amana_plot_no, 1),
            
            # ============ PAGE 3: Dimensions (plot width, plot length) ============
            ("page3_land_width", pdf_data.page3_land_width, ground_truth.land_width, 2),
            ("page3_land_length", pdf_data.page3_land_length, ground_truth.land_length, 2),
            
            # ============ PAGE 3 Footer: Areas table ============
            ("page3_footer_land_area", pdf_data.page3_footer_land_area, ground_truth.land_area, 2),
            ("page3_footer_gfa", pdf_data.page3_footer_gfa, ground_truth.gfa, 2),
            ("page3_footer_first_floor_area", pdf_data.page3_footer_first_floor_area, ground_truth.first_floor_area, 2),
            ("page3_footer_ground_floor_area", pdf_data.page3_footer_ground_floor_area, ground_truth.ground_floor_area, 2),
            ("page3_footer_ground_floor_annex", pdf_data.page3_footer_ground_floor_annex, ground_truth.ground_floor_annex, 2),
            ("page3_footer_second_floor_area", pdf_data.page3_footer_second_floor_area, ground_truth.second_floor_area, 2),
            
            # ============ PAGE 4 Header: Unit_Type, Ground Floor Area + Annex ============
            ("page4_header_unit_type", pdf_data.page4_header_unit_type, ground_truth.unit_type, 2),
            ("page4_header_area", pdf_data.page4_header_area, ground_truth.ground_floor_area + ground_truth.ground_floor_annex, 2),
            
            # ============ PAGE 5 Header: Unit_Type, First Floor Area ============
            ("page5_header_unit_type", pdf_data.page5_header_unit_type, ground_truth.unit_type, 2),
            ("page5_header_area", pdf_data.page5_header_area, ground_truth.first_floor_area, 2),
            
            # ============ PAGE 6 Header: Unit_Type, Second Floor Area ============
            ("page6_header_unit_type", pdf_data.page6_header_unit_type, ground_truth.unit_type, 2),
            ("page6_header_area", pdf_data.page6_header_area, ground_truth.second_floor_area, 2),
            
            # ============ PAGE 7 Header: Unit_Type (Roof) ============
            ("page7_header_unit_type", pdf_data.page7_header_unit_type, ground_truth.unit_type, 2),
            
            # ============ PAGE 8 Header Section 1: Unit_Type (Front Facade) ============
            ("page8_sec1_unit_type", pdf_data.page8_sec1_unit_type, ground_truth.unit_type, 2),
            
            # ============ PAGE 8 Header Section 2: Unit_Type (Back Facade) ============
            ("page8_sec2_unit_type", pdf_data.page8_sec2_unit_type, ground_truth.unit_type, 2),
            
            # ============ PAGE 9 Header: Unit_Type (Side Facade) ============
            ("page9_header_unit_type", pdf_data.page9_header_unit_type, ground_truth.unit_type, 2),
            
            # ============ CONSOLIDATED VALUES (kept for compatibility) ============
            # Critical checks
            ("unit_code", pdf_data.unit_code, ground_truth.unit_code, 1),
            ("plot_number", pdf_data.plot_number, ground_truth.amana_plot_no, 1),
            ("unit_type", pdf_data.unit_type, ground_truth.unit_type, 1),
            
            # High priority checks - Areas
            ("land_area", pdf_data.land_area, ground_truth.land_area, 2),
            ("ground_floor_area", pdf_data.ground_floor_area, ground_truth.ground_floor_area, 2),
            ("first_floor_area", pdf_data.first_floor_area, ground_truth.first_floor_area, 2),
            ("upper_floor_area", pdf_data.upper_floor_area, ground_truth.second_floor_area, 2),
            ("total_floor_area", pdf_data.total_floor_area, ground_truth.gfa, 2),
            ("ground_floor_annex", pdf_data.ground_floor_annex, ground_truth.ground_floor_annex, 2),
            
            # Medium priority checks - Dimensions (use page3 fields only, removed duplicates)
            # NOTE: land_length and land_width removed - use page3_land_length and page3_land_width instead
            ("street_width", pdf_data.street_width, ground_truth.street_width, 3),
            ("facade_type", pdf_data.facade_type, ground_truth.facade_type, 3),
            
            # Visual checks
            ("red_highlight_neighborhood", pdf_data.red_highlight_neighborhood, True, 2),
            ("red_highlight_plot", pdf_data.red_highlight_plot, True, 2),
        ]
        
        # Run all checks
        for field_name, pdf_val, gt_val, severity in checks:
            result = self._validate_field(field_name, pdf_val, gt_val, severity)
            report.all_results.append(result)
            report.total_checks += 1
            
            if result.status == ValidationStatus.MATCH:
                report.matches += 1
            elif result.status == ValidationStatus.MISMATCH:
                report.mismatches += 1
                self._add_to_failure_list(report, result)
            elif result.status == ValidationStatus.WARNING:
                report.warnings += 1
                self._add_to_failure_list(report, result)
            elif result.status == ValidationStatus.NOT_FOUND:
                report.not_found += 1
                self._add_to_failure_list(report, result)
        
        # Additional checks for header consistency
        self._validate_headers(pdf_data, ground_truth, report)
        
        # Determine overall status
        report.overall_status = self._determine_overall_status(report)
        report.summary = self._generate_summary(report)
        
        return report
    
    def _validate_field(self, field_name: str, pdf_val: Any, gt_val: Any, severity: int) -> ValidationResult:
        """Validate a single field"""
        result = ValidationResult(
            field_name=field_name,
            pdf_value=pdf_val,
            ground_truth_value=gt_val,
            status=ValidationStatus.MATCH,
            severity=severity
        )
        
        # Handle None/missing values
        if pdf_val is None or (isinstance(pdf_val, str) and pdf_val == ""):
            result.status = ValidationStatus.NOT_FOUND
            result.message = f"PDF value not found for {field_name}"
            return result
        
        if gt_val is None or (isinstance(gt_val, str) and gt_val == ""):
            result.status = ValidationStatus.WARNING
            result.message = f"Ground truth value not found for {field_name}"
            return result
        
        # Numeric comparison with tolerance
        if isinstance(pdf_val, (int, float)) and isinstance(gt_val, (int, float)):
            diff = abs(float(pdf_val) - float(gt_val))
            tolerance = self.tolerance
            
            # Use relative tolerance for larger values
            if abs(gt_val) > 10:
                tolerance = abs(gt_val) * 0.001  # 0.1% tolerance
            
            if diff <= tolerance:
                result.status = ValidationStatus.MATCH
                result.message = f"Values match within tolerance ({diff:.4f})"
            else:
                result.status = ValidationStatus.MISMATCH
                result.message = f"Value mismatch: PDF={pdf_val}, GT={gt_val}, diff={diff:.4f}"
        
        # String comparison
        elif isinstance(pdf_val, str) and isinstance(gt_val, str):
            if pdf_val.strip().upper() == gt_val.strip().upper():
                result.status = ValidationStatus.MATCH
            else:
                result.status = ValidationStatus.MISMATCH
                result.message = f"String mismatch: PDF='{pdf_val}', GT='{gt_val}'"
        
        # Boolean comparison
        elif isinstance(pdf_val, bool) and isinstance(gt_val, bool):
            if pdf_val == gt_val:
                result.status = ValidationStatus.MATCH
            else:
                result.status = ValidationStatus.MISMATCH
                result.message = f"Boolean mismatch: PDF={pdf_val}, GT={gt_val}"
        
        # Boolean vs expected True
        elif isinstance(pdf_val, bool) and gt_val is True:
            if pdf_val:
                result.status = ValidationStatus.MATCH
            else:
                result.status = ValidationStatus.MISMATCH
                result.message = f"Expected True but got False for {field_name}"
        
        # Type-coerced comparison
        else:
            if str(pdf_val).strip() == str(gt_val).strip():
                result.status = ValidationStatus.MATCH
            else:
                result.status = ValidationStatus.MISMATCH
                result.message = f"Value mismatch: PDF={pdf_val}, GT={gt_val}"
        
        return result
    
    def _add_to_failure_list(self, report: ValidationReport, result: ValidationResult):
        """Add result to appropriate failure list based on severity"""
        if result.severity == SEVERITY["CRITICAL"]:
            report.critical_failures.append(result)
        elif result.severity == SEVERITY["HIGH"]:
            report.high_failures.append(result)
        elif result.severity == SEVERITY["MEDIUM"]:
            report.medium_failures.append(result)
        else:
            report.low_failures.append(result)
    
    def _validate_headers(self, pdf_data: ExtractedData, ground_truth: GroundTruthRecord, report: ValidationReport):
        """Validate header consistency across pages"""
        logger.debug("Validating header consistency...")
        
        # Check that unit type is consistent across pages
        for page_num, header_info in pdf_data.header_info.items():
            unit_type = header_info.get("unit_type")
            if unit_type and unit_type != pdf_data.unit_type:
                result = ValidationResult(
                    field_name=f"header_unit_type_page_{page_num}",
                    pdf_value=unit_type,
                    ground_truth_value=pdf_data.unit_type,
                    status=ValidationStatus.WARNING,
                    severity=SEVERITY["MEDIUM"],
                    message=f"Header unit type inconsistent on page {page_num}",
                    page_number=page_num
                )
                report.all_results.append(result)
                report.warnings += 1
                report.total_checks += 1
    
    def _determine_overall_status(self, report: ValidationReport) -> str:
        """Determine overall validation status"""
        if len(report.critical_failures) > 0:
            return "CRITICAL_FAILURE"
        elif len(report.high_failures) > 0:
            return "HIGH_FAILURE"
        elif len(report.medium_failures) > 0:
            return "MEDIUM_FAILURE"
        elif report.mismatches > 0:
            return "PARTIAL_FAILURE"
        elif report.warnings > 0:
            return "PASSED_WITH_WARNINGS"
        else:
            return "PASSED"
    
    def _generate_summary(self, report: ValidationReport) -> str:
        """Generate a summary of the validation"""
        lines = [
            f"Validation Summary for {report.pdf_filename}",
            f"{'='*60}",
            f"Unit Code: {report.unit_code}",
            f"Plot Number: {report.plot_number}",
            f"",
            f"Total Checks: {report.total_checks}",
            f"  - Matches: {report.matches}",
            f"  - Mismatches: {report.mismatches}",
            f"  - Warnings: {report.warnings}",
            f"  - Not Found: {report.not_found}",
            f"",
            f"Overall Status: {report.overall_status}",
        ]
        
        if report.critical_failures:
            lines.append(f"\nCRITICAL FAILURES ({len(report.critical_failures)}):")
            for f in report.critical_failures:
                lines.append(f"  - {f.field_name}: {f.message}")
        
        if report.high_failures:
            lines.append(f"\nHIGH PRIORITY FAILURES ({len(report.high_failures)}):")
            for f in report.high_failures:
                lines.append(f"  - {f.field_name}: {f.message}")
        
        if report.medium_failures:
            lines.append(f"\nMEDIUM PRIORITY FAILURES ({len(report.medium_failures)}):")
            for f in report.medium_failures:
                lines.append(f"  - {f.field_name}: {f.message}")
        
        return "\n".join(lines)


def validate_pdf_against_ground_truth(
    pdf_data: ExtractedData, 
    ground_truth: GroundTruthRecord,
    tolerance: float = 0.01
) -> ValidationReport:
    """Convenience function for validation"""
    validator = Validator(tolerance)
    return validator.validate(pdf_data, ground_truth)


if __name__ == "__main__":
    from pdf_extractor import extract_pdf_data
    from excel_loader import GroundTruthLoader
    
    # Load ground truth
    loader = GroundTruthLoader("/mnt/user-data/uploads/ground_truth.xlsx")
    ground_truth_records = loader.load()
    
    # Extract PDF data
    pdf_data = extract_pdf_data("/mnt/user-data/uploads/DM1-D02-2A-25-453-01.pdf")
    
    # Get matching ground truth
    gt_record = ground_truth_records.get(pdf_data.plot_number)
    
    if gt_record:
        # Validate
        report = validate_pdf_against_ground_truth(pdf_data, gt_record)
        print(report.summary)
    else:
        print(f"No ground truth found for plot {pdf_data.plot_number}")
