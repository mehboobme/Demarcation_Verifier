"""
PDF Data Extractor Module - Enhanced Version v3
Extracts data from ROSHN property PDF documents using OpenAI GPT-4 Vision API.
"""

import re
import os
import cv2
import json
import ast
import base64
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pdfplumber
import fitz  # PyMuPDF for better text extraction
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import anthropic
from openai import OpenAI
import google.generativeai as genai
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys from environment variables
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractedData:
    """Container for all extracted PDF data"""
    filename: str = ""
    unit_code: str = ""
    plot_number: int = 0
    neighborhood: str = ""
    region: str = ""
    reference_map: str = ""
    unit_type: str = ""
    
    # Identifiers from Page 3
    nbhd_name: str = ""
    amana_block_no: int = 0
    phase: int = 0
    
    # Property type
    main_land_use: str = ""
    sub_land_use: str = ""
    attached_standalone: str = ""
    
    # Areas
    land_area: float = 0.0
    plot_extension: float = 0.0
    far: float = 0.0
    gfa: float = 0.0
    bua: float = 0.0
    ground_floor_area: float = 0.0
    ground_floor_annex: float = 0.0
    total_floor_area: float = 0.0
    first_floor_area: float = 0.0
    upper_floor_area: float = 0.0
    second_floor_area: float = 0.0  # Same as upper_floor_area, mapped separately
    
    # Dimensions
    land_length: float = 0.0
    land_width: float = 0.0
    land_shape: str = ""
    street_width: int = 0
    
    # Location attributes
    corner_unit: str = ""
    car_parking: int = 0
    orientation: str = ""
    front: str = ""
    rear: str = ""
    side: str = ""
    number_of_streets: int = 0
    no_of_facades: int = 0
    
    # Style
    facade_type: str = ""
    facade_color: int = 0
    
    # Visual checks
    red_highlight_neighborhood: bool = False
    red_highlight_plot: bool = False
    
    # ============ PAGE-SPECIFIC FIELDS (for validation report) ============
    # Page 1: Unit Code
    page1_unit_code: str = ""
    
    # Page 2 Section 1: NBHD_Name, AMANA_Plot_No
    page2_sec1_nbhd_name: str = ""
    page2_sec1_amana_plot_no: int = 0
    
    # Page 2 Section 2: NBHD_Name, Unit_Type, AMANA_Plot_No (from RED highlighted plot)
    page2_sec2_nbhd_name: str = ""
    page2_sec2_unit_type: str = ""
    page2_sec2_amana_plot_no: int = 0
    
    # Page 3: Dimensions (plot width, plot length)
    page3_land_width: float = 0.0
    page3_land_length: float = 0.0
    
    # Page 3 Footer: Areas table
    page3_footer_land_area: float = 0.0
    page3_footer_gfa: float = 0.0  # Total floor area (إجمالي مساحة الطوابق)
    page3_footer_first_floor_area: float = 0.0
    page3_footer_ground_floor_area: float = 0.0
    page3_footer_ground_floor_annex: float = 0.0
    page3_footer_second_floor_area: float = 0.0
    
    # Page 4 Header: Unit_Type, Ground Floor Area + Annex (calculated)
    page4_header_unit_type: str = ""
    page4_header_area: float = 0.0  # Ground_Floor_Area + Ground_Floor_Annex
    
    # Page 5 Header: Unit_Type, First Floor Area
    page5_header_unit_type: str = ""
    page5_header_area: float = 0.0  # First_Floor_Area
    
    # Page 6 Header: Unit_Type, Second Floor Area
    page6_header_unit_type: str = ""
    page6_header_area: float = 0.0  # Second_Floor_Area
    
    # Page 7 Header: Unit_Type (Roof/السطح)
    page7_header_unit_type: str = ""
    
    # Page 8 Header Section 1: Unit_Type (Front Facade - الواجهة الأمامية)
    page8_sec1_unit_type: str = ""
    
    # Page 8 Header Section 2: Unit_Type (Back Facade - الواجهة الخلفية)
    page8_sec2_unit_type: str = ""
    
    # Page 9 Header: Unit_Type (Side Facade - الواجهة الجانبية)
    page9_header_unit_type: str = ""
    
    # Header/Footer
    header_info: Dict[int, Dict] = field(default_factory=dict)
    footer_info: Dict[int, str] = field(default_factory=dict)
    
    # Debug
    raw_text_by_page: Dict[int, str] = field(default_factory=dict)
    extraction_notes: List[str] = field(default_factory=list)


class PDFExtractor:
    """Enhanced PDF extractor for ROSHN property documents with priority-based extraction"""
    
    # Standard dimensions by unit type (from ROSHN specifications)
    UNIT_DIMENSIONS = {
        'DA2': (25.14, 10.00, 15),
        'DA3': (25.00, 10.00, 15),
        'V1A': (25.11, 12.00, 15),
        'V2B': (25.13, 12.00, 15),
        'V2C': (25.13, 14.81, 15),
        'C20': (25.00, 16.00, 15),
        'VT1': (30.00, 18.40, 15),
        'VS1': (30.00, 27.74, 15),
        'DB2': (25.00, 12.00, 15),
        'DB3': (25.00, 11.54, 15),
    }
    
    def __init__(self, pdf_path: str, debug_dir: str = None, vision_model: str = "auto"):
        """
        Initialize PDF Extractor
        
        Args:
            pdf_path: Path to PDF file
            debug_dir: Directory for debug output
            vision_model: Vision model preference - "auto", "claude", "gemini"
                - "auto": Try text extraction first, then Claude, then Gemini
                - "claude": Use Claude Vision when vision is needed
                - "gemini": Use Gemini Vision when vision is needed
        """
        self.pdf_path = pdf_path
        self.debug_dir = debug_dir or "./debug_images"
        self.vision_model = vision_model  # User's preferred vision model
        self.gemini_quota_available = True  # Track Gemini quota status
        
        # Initialize API clients
        self.claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create subdirectory for this PDF
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.debug_subdir = os.path.join(self.debug_dir, pdf_name)
        os.makedirs(self.debug_subdir, exist_ok=True)
        
        self.data = ExtractedData()
        self.data.filename = os.path.basename(pdf_path)
        
        self.images: List[Image.Image] = []
        self.cv_images: List[np.ndarray] = []
        self.pdf_chars: Dict[int, List] = {}
    
    def _call_vision_model(self, image: Image.Image, extraction_type: str, 
                           prefer_model: str = None) -> Optional[Dict]:
        """
        Call vision model with priority-based fallback
        
        Priority: 
        1. User's preferred model (if specified)
        2. Claude Vision (reliable, paid)
        3. Gemini Vision (if quota available)
        
        Args:
            image: PIL Image to analyze
            extraction_type: Type of extraction (dimensions, facade, etc.)
            prefer_model: Override model preference for this call
        
        Returns:
            Extracted data dictionary or None
        """
        model_to_use = prefer_model or self.vision_model
        
        # If auto mode, try Claude first (more reliable)
        if model_to_use == "auto":
            model_to_use = "claude"
        
        result = None
        
        # Try primary model
        if model_to_use == "claude":
            logger.info(f"Calling Claude Vision for {extraction_type}...")
            result = self._extract_with_claude_vision(image, extraction_type)
            
            # If Claude fails and Gemini quota available, try Gemini as fallback
            if result is None and self.gemini_quota_available:
                logger.info(f"Claude failed, trying Gemini Vision for {extraction_type}...")
                result = self._extract_with_gemini_vision(image, extraction_type)
                
        elif model_to_use == "gemini":
            if self.gemini_quota_available:
                logger.info(f"Calling Gemini Vision for {extraction_type}...")
                result = self._extract_with_gemini_vision(image, extraction_type)
                
                # If Gemini fails (quota or other), try Claude as fallback
                if result is None:
                    logger.info(f"Gemini failed, trying Claude Vision for {extraction_type}...")
                    result = self._extract_with_claude_vision(image, extraction_type)
            else:
                # Gemini quota exhausted, use Claude
                logger.info(f"Gemini quota exhausted, using Claude Vision for {extraction_type}...")
                result = self._extract_with_claude_vision(image, extraction_type)
        
        return result
        
    def extract_all(self) -> ExtractedData:
        """Main extraction method"""
        logger.info(f"Starting extraction for: {self.pdf_path}")
        
        self._convert_to_images()
        self._extract_text_pdfplumber()
        self._extract_page1_unit_code()
        self._extract_page2_maps()
        self._extract_page3_site_plan()
        self._extract_dimensions_advanced()  # New advanced method
        self._extract_page4_ground_floor()
        self._extract_page5_first_floor()
        self._extract_page6_upper_floor()
        self._extract_page7_roof()
        self._extract_page8_facades()
        self._extract_page9_side_facade()
        self._extract_facade_style()
        self._detect_red_highlights()
        self._post_process()
        
        logger.info("Extraction complete")
        return self.data
    
    def _convert_to_images(self, dpi: int = 200):
        """Convert PDF pages to images"""
        logger.debug("Converting PDF to images...")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        poppler_path = os.path.join(script_dir, "poppler", "poppler-24.08.0", "Library", "bin")
        
        # Use local poppler if available, otherwise rely on system PATH
        if os.path.exists(poppler_path):
            self.images = convert_from_path(self.pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            self.images = convert_from_path(self.pdf_path, dpi=dpi)
            
        self.cv_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in self.images]
        
        for i, img in enumerate(self.images):
            img.save(os.path.join(self.debug_subdir, f"page_{i+1}.png"))
        
        logger.debug(f"Converted {len(self.images)} pages")
    
    def _extract_text_pdfplumber(self):
        """Extract text and character data using pdfplumber"""
        logger.debug("Extracting text with pdfplumber...")
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    self.data.raw_text_by_page[i+1] = text
                    self.pdf_chars[i+1] = page.chars
                    
                    with open(os.path.join(self.debug_subdir, f"page_{i+1}_text.txt"), "w", encoding="utf-8") as f:
                        f.write(text)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
    
    def _extract_page1_unit_code(self):
        """Extract unit code from page 1"""
        text = self.data.raw_text_by_page.get(1, "")
        
        pattern = r'(DM\d+-D\d+-\d[A-Z]-\d+-\d+)'
        match = re.search(pattern, text)
        
        if match:
            self.data.unit_code = match.group(1)
            self.data.page1_unit_code = match.group(1)  # Page-specific tracking
            logger.info(f"Unit code: {self.data.unit_code}")
            
            parts = self.data.unit_code.split('-')
            if len(parts) >= 5:
                self.data.plot_number = int(parts[4])
                self.data.neighborhood = parts[2]
    
    def _extract_page2_maps(self):
        """Extract map info from page 2 - Section 1 (overview) and Section 2 (detailed/highlighted)"""
        text = self.data.raw_text_by_page.get(2, "")
        
        # Extract reference map (unit type like DA2, V1A, etc.)
        ref_map_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        ref_maps = re.findall(ref_map_pattern, text)
        if ref_maps:
            self.data.reference_map = ref_maps[0]
        
        # Extract region (full format NB-2A)
        region_pattern = r'(NB-\d[A-Z])'
        match = re.search(region_pattern, text)
        if match:
            self.data.region = match.group(1)
        
        # ============ PAGE 2 SECTION 1 & 2: NBHD_Name ============
        # Extract NBHD name - only last 2 characters (e.g., "2A" from "NB-2A")
        nbhd_pattern = r'NB-(\d[A-Z])'
        nbhd_match = re.search(nbhd_pattern, text)
        if nbhd_match:
            nbhd_short = nbhd_match.group(1)  # Just "2A" without "NB-"
            self.data.page2_sec1_nbhd_name = nbhd_short
            self.data.page2_sec2_nbhd_name = nbhd_short
            self.data.nbhd_name = nbhd_short
            logger.info(f"NBHD Name from Page 2: {nbhd_short}")
        
        # ============ PAGE 2 SECTION 2: Unit_Type from text (الخريطة المرجعية) ============
        # Unit type appears next to الخريطة المرجعية in the text
        if ref_maps:
            self.data.page2_sec2_unit_type = ref_maps[0]
            logger.info(f"Unit Type from Page 2 text (الخريطة المرجعية): {ref_maps[0]}")
        
        # ============ PAGE 2 SECTION 2: AMANA_Plot_No from text (القطعة رقم) ============
        # Plot number appears in format "453 رقم القطعة" or similar
        plot_pattern = r'(\d{3})\s*(?:ﻢﻗﺭ|رقم)?\s*(?:ﺔﻌﻄﻘﻟﺍ|القطعة)?'
        plot_match = re.search(plot_pattern, text)
        if plot_match:
            plot_from_text = int(plot_match.group(1))
            self.data.page2_sec2_amana_plot_no = plot_from_text
            logger.info(f"Plot Number from Page 2 text (القطعة رقم): {plot_from_text}")
        
        # NOTE: Vision API verification for Page 2 was REMOVED because:
        # 1. Text extraction already works correctly for unit type and plot number
        # 2. Vision API was incorrectly reading plot 566 data (visible in neighborhood map)
        #    instead of the specific highlighted plot for each PDF
        # 3. This caused page2_sec2_unit_type and page2_sec2_amana_plot_no mismatches
        # Text extraction is reliable for these fields - no Vision API needed
    
    def _extract_page3_site_plan(self):
        """Extract site plan data from page 3"""
        text = self.data.raw_text_by_page.get(3, "")
        
        # Unit type
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        types = re.findall(type_pattern, text)
        if types:
            self.data.unit_type = types[0]
            logger.info(f"Unit type: {self.data.unit_type}")
        
        # Areas from table (page 3 footer - جدول المساحات)
        all_areas = re.findall(r'(\d+\.?\d*)\s*m2', text)
        logger.debug(f"Areas found: {all_areas}")
        
        if len(all_areas) >= 6:
            self.data.total_floor_area = float(all_areas[0])
            self.data.land_area = float(all_areas[1])
            self.data.first_floor_area = float(all_areas[2])
            self.data.ground_floor_area = float(all_areas[3])
            self.data.upper_floor_area = float(all_areas[4])
            self.data.second_floor_area = float(all_areas[4])  # Same as upper_floor
            self.data.ground_floor_annex = float(all_areas[5])
            
            # Page 3 footer specific fields
            self.data.page3_footer_gfa = float(all_areas[0])  # إجمالي مساحة الطوابق
            self.data.page3_footer_land_area = float(all_areas[1])  # مساحة الأرض
            self.data.page3_footer_first_floor_area = float(all_areas[2])  # مساحة الطابق الأول
            self.data.page3_footer_ground_floor_area = float(all_areas[3])  # مساحة الطابق الأرضي
            self.data.page3_footer_second_floor_area = float(all_areas[4])  # مساحة الطابق العلوي
            self.data.page3_footer_ground_floor_annex = float(all_areas[5])  # مساحة الملحق الأرضي
            
            logger.info(f"Areas - Land: {self.data.land_area}, GFA: {self.data.total_floor_area}")
    
    def _extract_dimensions_advanced(self):
        """
        Advanced dimension extraction with priority-based approach:
        1. First try text/regex extraction from page 3 (free, fast)
        2. If text extraction fails, use Vision API (Claude primary, Gemini fallback)
        """
        logger.info("Extracting dimensions...")
        
        # Use page 3 (site plan) for dimension extraction
        if len(self.images) < 3:
            logger.warning("Not enough pages for dimension extraction")
            return
        
        # ============ STEP 1: Try text extraction first (free) ============
        text_extraction_success = self._try_text_dimension_extraction()
        
        if text_extraction_success:
            logger.info("Dimensions extracted successfully via text extraction (no Vision API needed)")
            return
        
        # ============ STEP 2: Text extraction failed, use Vision API ============
        logger.info("Text extraction insufficient, using Vision API for dimensions...")
        
        try:
            # Get page 3 image
            page3_img = self.images[2]  # 0-indexed
            
            # Use priority-based vision model call
            dimensions = self._call_vision_model(page3_img, "dimensions")
            
            if dimensions:
                # Verify that Vision API found the correct plot
                plot_found = str(dimensions.get('plot_found', ''))
                expected_plot = str(self.data.plot_number)
                
                if plot_found == 'NOT_FOUND':
                    logger.warning(f"Vision API could not find plot {expected_plot} on page 3")
                    self._fallback_to_defaults()
                    return
                    
                if plot_found and plot_found != expected_plot:
                    logger.warning(f"Vision API found wrong plot: {plot_found} (expected {expected_plot}), using fallback")
                    self._fallback_to_defaults()
                    return
                
                logger.info(f"Vision API confirmed plot found: {plot_found}")
                
                # STEP 1: Handle trapezoid plots FIRST - always use max of front/back widths
                # This must be done before processing land_width to ensure correct value
                final_width = 0.0
                width_front = 0.0
                width_back = 0.0
                
                if 'width_front' in dimensions and dimensions['width_front']:
                    width_front = float(dimensions['width_front'])
                if 'width_back' in dimensions and dimensions['width_back']:
                    width_back = float(dimensions['width_back'])
                
                # For trapezoid plots, use max(front, back) for the width
                if width_front > 0 and width_back > 0:
                    final_width = max(width_front, width_back)
                    logger.info(f"Trapezoid plot - front: {width_front}, back: {width_back}, using MAX: {final_width}")
                elif 'land_width' in dimensions and dimensions['land_width']:
                    final_width = float(dimensions['land_width'])
                
                # Validate and set width
                if 9.0 <= final_width <= 30.0:
                    self.data.land_width = final_width
                    self.data.page3_land_width = final_width
                    logger.info(f"Land width from Vision API: {self.data.land_width}")
                else:
                    logger.warning(f"Invalid land_width {final_width}, will use fallback")
                
                # STEP 2: Handle land length
                if 'land_length' in dimensions and dimensions['land_length']:
                    extracted_length = float(dimensions['land_length'])
                    # Validate extracted length is in expected range (25-30m)
                    if 24.0 <= extracted_length <= 31.0:
                        self.data.land_length = extracted_length
                        self.data.page3_land_length = extracted_length
                        logger.info(f"Land length from Vision API: {self.data.land_length}")
                    else:
                        logger.warning(f"Invalid land_length {extracted_length}, will use fallback")
                
                # Street width - extract from Vision API
                if 'street_width' in dimensions and dimensions['street_width']:
                    street_w = int(dimensions['street_width'])
                    self.data.street_width = street_w
                else:
                    self.data.street_width = 15  # Default only if Vision fails to extract
                logger.info(f"Street width: {self.data.street_width}")
            
            # Call fallback for any missing values (if Vision didn't extract properly)
            if self.data.land_length == 0.0 or self.data.land_width == 0.0:
                logger.warning("Vision API did not extract all dimensions, using fallback")
                self._fallback_to_defaults()
                    
        except Exception as e:
            logger.error(f"Vision dimension extraction failed: {e}")
            # Fallback to unit type defaults
            self._fallback_to_defaults()
    
    def _try_text_dimension_extraction(self) -> bool:
        """
        Try to extract dimensions from page 3 using PyMuPDF text extraction.
        This analyzes text positions to find dimensions associated with our plot.
        
        Returns:
            True if dimensions were successfully extracted, False otherwise
        """
        try:
            return self._extract_dimensions_pymupdf()
        except Exception as e:
            logger.debug(f"PyMuPDF dimension extraction failed: {e}")
            return False
    
    def _extract_dimensions_pymupdf(self) -> bool:
        """
        Extract dimensions using PyMuPDF by:
        1. Finding the plot number position on page 3
        2. Finding dimension values near that position
        3. Determining which is width and which is length based on position/context
        
        Returns:
            True if dimensions were successfully extracted, False otherwise
        """
        doc = fitz.open(self.pdf_path)
        if len(doc) < 3:
            doc.close()
            return False
        
        page = doc[2]  # Page 3 (0-indexed)
        plot_num = str(self.data.plot_number)
        
        text_dict = page.get_text('dict')
        
        # Step 1: Find our plot number position
        plot_position = None
        unit_type_position = None
        
        for block in text_dict['blocks']:
            if 'lines' not in block:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    bbox = span['bbox']
                    
                    if text == plot_num:
                        plot_position = (bbox[0], bbox[1])
                        logger.debug(f"Found plot {plot_num} at position {plot_position}")
                    
                    # Also find unit type position (should be near plot)
                    if text == self.data.unit_type:
                        unit_type_position = (bbox[0], bbox[1])
        
        if not plot_position:
            logger.debug(f"Plot {plot_num} not found on page 3")
            doc.close()
            return False
        
        # Step 2: Collect all dimension values with their positions
        dimensions = []
        for block in text_dict['blocks']:
            if 'lines' not in block:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    bbox = span['bbox']
                    
                    # Match dimension patterns (XX.XX format)
                    if re.match(r'^\d{1,2}\.\d{2}$', text):
                        value = float(text)
                        x, y = bbox[0], bbox[1]
                        
                        # Calculate distance from plot position
                        dist = ((x - plot_position[0])**2 + (y - plot_position[1])**2)**0.5
                        dimensions.append({
                            'value': value,
                            'x': x,
                            'y': y,
                            'distance': dist
                        })
        
        # Sort by distance from plot
        dimensions.sort(key=lambda x: x['distance'])
        
        # Step 3: Identify width and length from nearby dimensions
        # Width values are typically 9-28m range (up to 28 for large VS1 plots)
        # Length values are typically 24-31m range
        # Note: For large plots, width can be > 24m, so we need special handling
        
        width_candidates = []
        length_candidates = []
        all_candidates = []  # Store all candidates for special cases
        
        for dim in dimensions[:25]:  # Check closest 25 dimensions
            val = dim['value']
            all_candidates.append(dim)
            if 9.0 <= val < 24.0:  # Normal width range
                width_candidates.append(dim)
            elif 24.0 <= val <= 31.0:  # Length range (may include large widths)
                length_candidates.append(dim)
        
        # For width: if two candidates are close in distance, pick the LARGER value
        # This handles trapezoid plots where the larger width is the street frontage
        extracted_width = None
        extracted_length = None
        
        if width_candidates:
            # Get closest candidate
            closest = width_candidates[0]
            # Look for other candidates within 15 units distance (increased for irregular plots)
            nearby = [w for w in width_candidates if w['distance'] <= closest['distance'] + 15]
            # Pick the largest value among nearby candidates
            best = max(nearby, key=lambda x: x['value'])
            extracted_width = best['value']
            logger.info(f"PyMuPDF: Found width {extracted_width} at distance {best['distance']:.1f} (from {len(nearby)} nearby candidates)")
        
        if length_candidates:
            # For length, take the largest nearby value (this is the actual length)
            closest_len = length_candidates[0]
            # Expand search to 50 units for large plots that may have width in length range
            nearby_lens = [l for l in length_candidates if l['distance'] <= closest_len['distance'] + 50]
            best_len = max(nearby_lens, key=lambda x: x['value'])
            extracted_length = best_len['value']
            logger.info(f"PyMuPDF: Found length {extracted_length} at distance {best_len['distance']:.1f}")
            
            # Special case: For large plots (VS1, VT1), width might be in the "length" range
            # If we have multiple values in length range, check if one is actually width
            if len(nearby_lens) >= 2:
                sorted_lens = sorted(nearby_lens, key=lambda x: x['value'])
                smaller_len = sorted_lens[0]['value']
                larger_len = sorted_lens[-1]['value']
                
                # Check if smaller "length" should actually be the width
                # This applies when: smaller < 28 and larger >= 30 (standard length)
                if smaller_len < 28.0 and larger_len >= 30.0:
                    # If smaller_len is larger than current width, use it as width
                    if extracted_width is None or smaller_len > extracted_width:
                        logger.info(f"PyMuPDF: Large plot detected - smaller_len={smaller_len}, larger_len={larger_len}, current_width={extracted_width}")
                        extracted_width = smaller_len
                        extracted_length = larger_len
                        logger.info(f"PyMuPDF: Large plot - width={extracted_width}, length={extracted_length}")
        
        # Step 4: Extract street width from text
        full_text = page.get_text()
        street_width = self._extract_street_width_from_text(full_text)
        
        doc.close()
        
        # Step 5: Validate and set values
        if extracted_width and extracted_length:
            self.data.land_width = extracted_width
            self.data.page3_land_width = extracted_width
            self.data.land_length = extracted_length
            self.data.page3_land_length = extracted_length
            
            if street_width:
                self.data.street_width = street_width
            
            logger.info(f"PyMuPDF extraction successful: width={extracted_width}, length={extracted_length}, street={self.data.street_width}")
            return True
        
        return False
    
    def _extract_street_width_from_text(self, text: str) -> Optional[int]:
        """
        Extract street width from Arabic text on page 3.
        Most ROSHN residential plots have 15m streets.
        Returns the most common street width found, defaulting to 15.
        """
        # Find all street width values
        street_widths = []
        
        # Pattern for 'شارع عرض' followed by number
        patterns = [
            r'ﺷﺎﺭﻉ\s*ﻋﺮﺽ\s*(\d{1,2})',
            r'شارع\s*عرض\s*(\d{1,2})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            street_widths.extend([int(m) for m in matches])
        
        # Check for inline patterns
        if 'ﺷﺎﺭﻉ ﻋﺮﺽ15' in text or 'شارع عرض15' in text:
            street_widths.append(15)
        if 'ﺷﺎﺭﻉ ﻋﺮﺽ16' in text or 'شارع عرض16' in text:
            street_widths.append(16)
        
        if not street_widths:
            return None
        
        # Return the most common street width (usually 15 for residential)
        # Filter to valid values (15 or 16 for ROSHN)
        valid_widths = [w for w in street_widths if w in [15, 16]]
        if valid_widths:
            # Count occurrences and return most common
            from collections import Counter
            return Counter(valid_widths).most_common(1)[0][0]
        
        return 15  # Default for ROSHN residential
    
    def _fallback_to_defaults(self):
        """Fallback to unit type defaults if vision extraction fails"""
        if self.data.unit_type in self.UNIT_DIMENSIONS:
            defaults = self.UNIT_DIMENSIONS[self.data.unit_type]
            
            if self.data.land_length == 0.0:
                self.data.land_length = defaults[0]
                self.data.extraction_notes.append(f"Land length from unit type default: {defaults[0]}")
                logger.info(f"Land length from unit type: {self.data.land_length}")
            
            if self.data.land_width == 0.0:
                self.data.land_width = defaults[1]
                self.data.extraction_notes.append(f"Land width from unit type default: {defaults[1]}")
                logger.info(f"Land width from unit type: {self.data.land_width}")
            
            if self.data.street_width == 0:
                self.data.street_width = defaults[2]
                self.data.extraction_notes.append(f"Street width from unit type default: {defaults[2]}")
                logger.info(f"Street width from unit type: {self.data.street_width}")
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _extract_with_claude_vision(self, img: Image.Image, extraction_type: str) -> dict:
        """Use Claude Vision to extract data from image"""
        
        img_base64 = self._image_to_base64(img)
        
        if extraction_type == "dimensions":
            # Include plot number and unit type in prompt for better targeting
            plot_num = str(self.data.plot_number) if self.data.plot_number else ""
            unit_type = self.data.unit_type if self.data.unit_type else ""
            
            # Width hints based on unit type
            width_hints = {
                'DA2': '10.00', 'DA3': '10.00', 'V1A': '12.00', 'V2B': '12.00 or 12.55',
                'V2C': '14.00 or 14.81', 'C20': '16.00, 16.74, or 20.22', 'VT1': '18.40', 'VS1': '27.74', 'DB3': '11.54'
            }
            expected_width = width_hints.get(unit_type, '10-28')
            
            prompt = f"""You are analyzing a ROSHN real estate site plan (Page 3). This page shows MANY plots with dimension labels.

**CRITICAL TASK: Find plot "{plot_num}" with "{unit_type}" written inside, then read ALL dimension numbers EXACTLY as written.**

**STEP 1: LOCATE THE CORRECT PLOT**
- Find the plot shape that has BOTH "{plot_num}" AND "{unit_type}" written INSIDE it
- These labels appear together inside the plot boundary
- IGNORE other plots with different numbers

**STEP 2: READ DIMENSION NUMBERS WITH EXACT DECIMAL PRECISION**

**CRITICAL: Read the EXACT numbers as written - do NOT round or approximate!**
- "12.55" must be reported as 12.55 (NOT 12.0)
- "20.22" must be reported as 20.22 (NOT 20.0)
- "14.81" must be reported as 14.81 (NOT 15.0 or 14.0)
- "11.54" must be reported as 11.54 (NOT 11.5 or 12.0)
- "25.14" must be reported as 25.14 (NOT 25.0)

**LAND LENGTH (TOP and BOTTOM edges):** 
- Read the EXACT dimension written along horizontal edges
- Common EXACT values: 25.00, 25.11, 25.13, 25.14, 30.00

**LAND WIDTH (FRONT and BACK edges) - CRITICAL RULES:**

**RULE 1: MULTIPLE DIMENSIONS ON SAME EDGE**
- When you see TWO dimension numbers near the same edge (e.g., 12.02 and 15.64)
- Use the dimension that is INSIDE the plot boundary (the plot's actual wall line)
- IGNORE the dimension that is OUTSIDE the boundary (belongs to adjacent area)
- Example: Back edge shows 12.02 (inside) and 15.64 (outside) → Use 12.02

**RULE 2: FRONT vs BACK COMPARISON**  
- width_front = dimension on STREET-FACING edge (front of plot)
- width_back = dimension INSIDE the plot boundary on rear edge
- land_width = MAXIMUM of (width_front, width_back)

**EXAMPLE - Plot 779 (V2C):**
- Front edge (street side): 14.81
- Back edge: TWO numbers visible - 12.02 (inside boundary) and 15.64 (outside boundary)
- Use 12.02 for back (inside the plot wall)
- width_front=14.81, width_back=12.02
- land_width = max(14.81, 12.02) = **14.81**

**EXACT DIMENSION VALUES THAT EXIST IN ROSHN PLOTS:**
Lengths: 25.00, 25.11, 25.13, 25.14, 30.00
Widths: 10.00, 10.77, 11.54, 12.00, 12.02, **12.55**, 13.41, 14.00, **14.81**, 15.64, 16.00, **16.74**, 18.40, **20.22**, 27.74

**MORE EXAMPLES:**

Plot 578 (V2B): front=12.00, back=12.55 (both inside boundary)
→ land_width = max(12.00, 12.55) = **12.55**

Plot 566 (C20): front=16.74, back=20.22 (both inside boundary)
→ land_width = max(16.74, 20.22) = **20.22**

Plot 554 (DB3): front=10.77, back=11.54 (both inside boundary)
→ land_width = max(10.77, 11.54) = **11.54**

**RETURN FORMAT (JSON only, EXACT decimal values):**
{{
  "plot_found": "{plot_num}",
  "land_length": 25.00,
  "land_width": 14.81,
  "width_front": 14.81,
  "width_back": 12.02,
  "street_width": 15
}}

If plot "{plot_num}" with "{unit_type}" cannot be found:
{{"plot_found": "NOT_FOUND", "land_length": 0, "land_width": 0, "street_width": 15}}"""
        
        elif extraction_type == "facade":
            prompt = """You are analyzing building elevation/facade images from a ROSHN property document.

**TASK: Determine if this is TRADITIONAL (T) or MODERN (M) architectural style.**

**THE KEY INDICATOR: HALF CIRCLES (ARCHES) ABOVE DOORS**

**TRADITIONAL STYLE (T):**
- Look for HALF CIRCLE / SEMI-CIRCULAR ARCH above ANY door
- The arch appears as a curved/rounded shape ABOVE the door frame
- If you find even ONE door with a half circle arch above it → TRADITIONAL (T)
- The arch is clearly visible as a curved line forming a semi-circle over the door opening

**MODERN STYLE (M):**
- ALL doors have FLAT/STRAIGHT tops (rectangular openings)
- NO half circles or arches above any doors
- Door panels may have VERTICAL LINES/SLATS pattern
- Clean geometric lines, completely rectangular shapes

**DECISION RULE:**
1. Scan ALL doors visible in the elevation image
2. Look for ANY half circle / arch shape above the doors
3. Found at least ONE half circle arch above a door? → **TRADITIONAL (T)**
4. NO half circles found above any door? → **MODERN (M)**

**VISUAL CLUE:**
- TRADITIONAL: Half circle (⌒) shape clearly visible above door frame
- MODERN: Flat horizontal line (—) above door frame, no curves

Return ONLY valid JSON:
{{"facade_type": "T"}} if ANY door has a half circle arch above it
{{"facade_type": "M"}} if NO doors have arches (all flat tops)"""
        
        elif extraction_type == "page2_verification":
            prompt = """Analyze Page 2 of this ROSHN property document.

**TASK: Find the RED HIGHLIGHTED plot in SECTION 2 (the detailed zoomed map) and extract:**

1. **Plot Number**: The 3-digit number written INSIDE the RED colored plot (e.g., 453, 566, 740, 741, 779)
2. **Unit Type**: The villa type code written INSIDE the RED colored plot:
   - DA2, DA3, DB2, DB3 (Duplex types)
   - V1A, V2B, V2C (Villa types)  
   - VT1, VS1 (Villa Twin/Special)
   - C20 (Corner)

**HOW TO IDENTIFY:**
- Look for the plot filled with RED/PINK color in Section 2
- The plot number and villa type are written at the CENTER of the red area
- IGNORE all white/uncolored neighboring plots

Return ONLY valid JSON:
{{"plot_number": 566, "unit_type": "C20"}}"""
        
        else:
            return {}
        
        try:
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            response_text = message.content[0].text.strip()
            logger.debug(f"Claude Vision response: {response_text}")
            
            # Parse JSON from response
            # Handle cases where response might have extra text
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Claude Vision API error: {e}")
            return {}
    
    def _extract_with_gemini_vision(self, img: Image.Image, extraction_type: str) -> dict:
        """Use Google Gemini Vision to extract data from image"""
        
        if extraction_type == "dimensions":
            plot_num = str(self.data.plot_number) if self.data.plot_number else ""
            unit_type = self.data.unit_type if self.data.unit_type else ""
            
            # Width hints based on unit type
            width_hints = {
                'DA2': '10.00', 'DA3': '10.00', 'V1A': '12.00', 'V2B': '12.00-12.55',
                'V2C': '14.81', 'C20': '16.00-20.22', 'VT1': '18.40', 'VS1': '27.74', 'DB3': '11.54'
            }
            expected_width = width_hints.get(unit_type, '10-20')
            
            prompt = f"""CRITICAL TASK: Find the plot with BOTH "{plot_num}" AND "{unit_type}" written INSIDE it.

**WARNING: This page shows MANY plots. You MUST find the plot that has BOTH the number "{plot_num}" AND villa type "{unit_type}" written INSIDE.**

**STEP 1 - LOCATE THE CORRECT PLOT:**
- Scan the ENTIRE image for a plot that contains BOTH:
  1. The number "{plot_num}" written inside
  2. The villa type "{unit_type}" written inside (near the plot number)
- The plot number and villa type should appear TOGETHER inside the same plot boundary
- IGNORE plots that show different numbers (566, 565, 567, etc.)
- IGNORE plots that only show the number without the villa type

**STEP 2 - READ DIMENSIONS FROM THE PLOT EDGES:**
Once you find the plot with "{plot_num}" + "{unit_type}" inside:

- LAND LENGTH: The longer dimension (horizontal), typically 25.00-30.00m
  Read from the edge parallel to the street
  
- LAND WIDTH: The shorter dimension (perpendicular to street)
  For unit type {unit_type}, expected width is approximately {expected_width} meters
  Read from BOTH the LEFT and RIGHT edges of the plot
  **IF TRAPEZOID (left ≠ right): Return the LARGER value as land_width**
  Also report both values separately as width_front and width_back

**STEP 3 - VERIFY:**
- Confirm "{plot_num}" AND "{unit_type}" are BOTH inside the plot you measured
- Width should be exact match to {expected_width}m
- If width is 20.22 and unit type is NOT C20, you found the WRONG plot

**RETURN FORMAT (JSON only):**
{{"plot_found": "{plot_num}", "land_length": 25.00, "land_width": 10.00, "width_front": 10.00, "width_back": 10.00, "street_width": 15}}

If you cannot find plot {plot_num} with {unit_type} inside: 
{{"plot_found": "NOT_FOUND", "land_length": 0, "land_width": 0, "street_width": 15}}"""

        elif extraction_type == "facade":
            prompt = """You are analyzing building elevation/facade images from a ROSHN property document.

**TASK: Determine if this is TRADITIONAL (T) or MODERN (M) architectural style.**

**THE KEY INDICATOR: HALF CIRCLES (ARCHES) ABOVE DOORS**

Scan ALL doors in this facade image (main door, side doors, garage doors, any door).

**TRADITIONAL STYLE (T):**
- Look for HALF-CIRCLE or ARCH shapes above ANY door
- The arch is a curved/semicircular decorative element above the door frame
- Even if just ONE door has a half-circle arch above it → TRADITIONAL

**MODERN STYLE (M):**
- NO half-circles or arches above ANY doors
- All doors have completely FLAT/STRAIGHT tops (rectangular)
- Vertical line/slat patterns on doors with flat tops
- Clean geometric lines, no curved elements

**DECISION RULE:**
1. Scan EVERY door in the image
2. Found ANY door with a half-circle/arch above it? → TRADITIONAL (T)
3. ALL doors have flat tops with NO arches? → MODERN (M)

**IMPORTANT:**
- Even ONE door with an arch = TRADITIONAL
- The arch is above the door, creating a curved silhouette
- Ignore windows, focus on DOORS

Return ONLY valid JSON (no explanation, no markdown):
{{"facade_type": "T"}} for Traditional (any door has arch above it)
{{"facade_type": "M"}} for Modern (no arches above any door)"""

        elif extraction_type == "page2_verification":
            prompt = """Analyze Page 2 of this ROSHN property document.

**TASK: Find the RED HIGHLIGHTED plot in SECTION 2 (the detailed zoomed map) and extract:**

1. **Plot Number**: The 3-digit number written INSIDE the RED colored plot (e.g., 453, 566, 740, 741, 779)
2. **Unit Type**: The villa type code written INSIDE the RED colored plot:
   - DA2, DA3, DB2, DB3 (Duplex types)
   - V1A, V2B, V2C (Villa types)  
   - VT1, VS1 (Villa Twin/Special)
   - C20 (Corner)

**HOW TO IDENTIFY:**
- Look for the plot filled with RED/PINK color in Section 2
- The plot number and villa type are written at the CENTER of the red area
- IGNORE all white/uncolored neighboring plots

Return ONLY valid JSON (no explanation, no markdown):
{{"plot_number": 566, "unit_type": "C20"}}"""

        else:
            return {}
        
        try:
            response = self.gemini_model.generate_content([prompt, img])
            response_text = response.text.strip()
            logger.debug(f"Gemini Vision response: {response_text}")
            
            # Parse JSON from response - handle various formats
            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                json_str = json_match.group()
                
                # Try standard JSON first
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                
                # Try with single quotes replaced
                try:
                    json_str_fixed = json_str.replace("'", '"')
                    return json.loads(json_str_fixed)
                except json.JSONDecodeError:
                    pass
                
                # Try ast.literal_eval for Python dict syntax
                try:
                    result = ast.literal_eval(json_str)
                    if isinstance(result, dict):
                        return result
                except (ValueError, SyntaxError):
                    pass
            
            # Last resort: try to parse the whole response
            try:
                return json.loads(response_text.replace("'", '"'))
            except json.JSONDecodeError:
                pass
            
            try:
                result = ast.literal_eval(response_text)
                if isinstance(result, dict):
                    return result
            except (ValueError, SyntaxError):
                pass
            
            logger.error(f"Could not parse Gemini response as JSON: {response_text}")
            return {}
            
        except Exception as e:
            logger.error(f"Gemini Vision API error: {e}")
            return {}

    def _extract_with_openai_vision(self, img: Image.Image, extraction_type: str) -> dict:
        """Use OpenAI GPT-4 Vision to extract data from image"""
        
        img_base64 = self._image_to_base64(img)
        
        if extraction_type == "dimensions":
            plot_num = str(self.data.plot_number) if self.data.plot_number else ""
            unit_type = self.data.unit_type if self.data.unit_type else ""
            prompt = f"""You are analyzing a ROSHN real estate site plan (Page 3). This document shows multiple plots with dimensions.

**YOUR TASK: Find Plot Number "{plot_num}" and extract its dimensions.**

**STEP 1: SEARCH FOR PLOT NUMBER "{plot_num}"**
- Look for the number "{plot_num}" written on this page
- It may be written inside a plot area along with villa type "{unit_type}"
- The plot may be highlighted in red/pink color OR just have the number written on it
- IGNORE all other plots - you MUST find and measure plot "{plot_num}" specifically
- Do NOT read dimensions from neighboring plots like 566, 740, etc.

**STEP 2: Once you find plot "{plot_num}", READ ITS BOUNDARY DIMENSIONS**
Look at the numbers written along EACH EDGE of plot "{plot_num}":

**LAND LENGTH (horizontal dimension - typically longer):**
- Read from TOP or BOTTOM edge of the plot
- Expected values: 25.00-30.00 meters (typical: 25.11, 25.13, 25.14, 30.00)

**LAND WIDTH (vertical dimension - perpendicular to street):**
- Read from LEFT side and RIGHT side edges of plot "{plot_num}"
- Villa type "{unit_type}" typically has width: DA2=10m, V1A/V2B=12m, V2C=14.81m, C20=16-20m, VT1=18.40m, VS1=27.74m
- **IF LEFT ≠ RIGHT (trapezoid): Return the LARGER value**
  Example: Left=10.00m, Right=12.00m → return 12.00

**IMPORTANT VERIFICATION:**
- Plot "{plot_num}" with type "{unit_type}" should have width close to the expected value for that villa type
- If you're reading width > 15m for DA2/V1A/V2B type plots, you're likely reading the WRONG plot

Return ONLY valid JSON:
{{"land_length": 25.14, "land_width": 10.00, "street_width": 15}}"""

        elif extraction_type == "facade":
            prompt = """You are analyzing building elevation/facade images from a ROSHN property document.

**TASK: Determine if this is TRADITIONAL (T) or MODERN (M) architectural style.**

**THE KEY INDICATOR: HALF CIRCLES (ARCHES) ABOVE DOORS**

Scan ALL doors in this facade image (main door, side doors, garage doors, any door).

**TRADITIONAL STYLE (T):**
- Look for HALF-CIRCLE or ARCH shapes above ANY door
- The arch is a curved/semicircular decorative element above the door frame
- Even if just ONE door has a half-circle arch above it → TRADITIONAL

**MODERN STYLE (M):**
- NO half-circles or arches above ANY doors
- All doors have completely FLAT/STRAIGHT tops (rectangular)
- Vertical line/slat patterns on doors with flat tops
- Clean geometric lines, no curved elements

**DECISION RULE:**
1. Scan EVERY door in the image
2. Found ANY door with a half-circle/arch above it? → TRADITIONAL (T)
3. ALL doors have flat tops with NO arches? → MODERN (M)

**IMPORTANT:**
- Even ONE door with an arch = TRADITIONAL
- The arch is above the door, creating a curved silhouette
- Ignore windows, focus on DOORS

Return ONLY valid JSON:
{{"facade_type": "T"}} or {{"facade_type": "M"}}"""

        elif extraction_type == "page2_verification":
            prompt = """Analyze Page 2 of this ROSHN property document.

**TASK: Find the RED HIGHLIGHTED plot in SECTION 2 (the detailed map) and extract:**

1. **Plot Number**: The 3-digit number written INSIDE the RED colored plot
2. **Unit Type**: The villa type code written INSIDE the RED colored plot

Return ONLY valid JSON:
{{"plot_number": 566, "unit_type": "C20"}}"""

        else:
            return {}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ],
                    }
                ],
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI Vision response: {response_text}")
            
            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            return {}

    def _extract_dimensions_near_plot(self, chars: List, plot_num: str):
        """Find dimensions near the plot number in the site plan"""
        logger.debug(f"Looking for dimensions near plot {plot_num}")
        
        # Find position of plot number
        plot_positions = []
        for i in range(len(chars) - len(plot_num) + 1):
            seq = ''.join([chars[j]['text'] for j in range(i, i + len(plot_num))])
            if seq == plot_num:
                x = chars[i]['x0']
                y = chars[i]['top']
                plot_positions.append((x, y))
        
        if not plot_positions:
            return
        
        # For each plot position, find nearby dimension-like numbers
        for px, py in plot_positions:
            # Get characters within 200 pixels
            nearby = [c for c in chars if abs(c['x0'] - px) < 200 and abs(c['top'] - py) < 150]
            nearby.sort(key=lambda c: (round(c['top']/5), c['x0']))
            
            nearby_text = ''.join([c['text'] for c in nearby])
            logger.debug(f"Text near plot at ({px:.0f}, {py:.0f}): {nearby_text[:100]}")
            
            # Look for 25.XX pattern (length)
            length_match = re.search(r'25[\.](\d{2})', nearby_text)
            if length_match and self.data.land_length == 0.0:
                self.data.land_length = float(f"25.{length_match.group(1)}")
                logger.info(f"Land length from nearby: {self.data.land_length}")
            
            # Look for 10.XX pattern (width)
            width_match = re.search(r'10[\.](\d{1,2})', nearby_text)
            if width_match and self.data.land_width == 0.0:
                self.data.land_width = float(f"10.{width_match.group(1)[:2]}")
                logger.info(f"Land width from nearby: {self.data.land_width}")
    
    def _extract_street_width_advanced(self, full_text: str, chars: List):
        """Extract street width using multiple methods"""
        logger.debug("Extracting street width...")
        
        # Method 1: "15ﺮﺘﻣ" or "15 متر" (15 meters)
        if re.search(r'15\s*(?:ﺮﺘﻣ|متر|m)', full_text):
            self.data.street_width = 15
            logger.info("Street width from '15m' pattern: 15")
            return
        
        # Method 2: Look for "15" near "شارع" (street) or "عرض" (width)
        street_patterns = [
            r'(?:شارع|ﺭﺎﺷ|عرض|ﺽﺮﻋ).*?(\d{1,2})',
            r'(\d{1,2}).*?(?:شارع|ﺭﺎﺷ|عرض|ﺽﺮﻋ)',
        ]
        
        for pattern in street_patterns:
            match = re.search(pattern, full_text)
            if match:
                width = int(match.group(1))
                if 10 <= width <= 30:
                    self.data.street_width = width
                    logger.info(f"Street width from street pattern: {width}")
                    return
        
        # Method 3: Find "15" in site plan area (not in table)
        site_plan_chars = [c for c in chars if 200 < c['top'] < 1000]
        site_text = ''.join([c['text'] for c in sorted(site_plan_chars, key=lambda c: (c['top'], c['x0']))])
        
        # Look for isolated "15" or "15." pattern
        if re.search(r'(?:^|[^\d])15(?:[^\d\.]|$)', site_text):
            self.data.street_width = 15
            logger.info("Street width from site plan '15': 15")
            return
        
        # Method 4: Context-based detection
        # Group chars by Y position to find street annotations
        y_groups = {}
        for c in chars:
            y_key = round(c['top'] / 20) * 20
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(c)
        
        for y_key in sorted(y_groups.keys()):
            line_chars = sorted(y_groups[y_key], key=lambda c: c['x0'])
            line_text = ''.join([c['text'] for c in line_chars])
            
            # Check if this line has street-related content with "15"
            if ('15' in line_text and 
                any(x in line_text for x in ['ﺮﺘﻣ', 'متر', 'm', 'ﺭﺎﺷ', 'شارع'])):
                self.data.street_width = 15
                logger.info(f"Street width from line analysis: 15")
                return
    
    def _extract_page4_ground_floor(self):
        """Extract ground floor area and unit type from page 4 header"""
        text = self.data.raw_text_by_page.get(4, "")
        
        # Extract area from header (first area value in format XXX.XX m²)
        area_pattern = r'(\d+\.?\d*)\s*m[²2]?'
        match = re.search(area_pattern, text)
        if match:
            area_val = float(match.group(1))
            if self.data.ground_floor_area == 0.0:
                self.data.ground_floor_area = area_val
            # Page 4 header area = Ground_Floor_Area + Ground_Floor_Annex
            self.data.page4_header_area = area_val
        
        # Extract unit type from page 4 header (e.g., "دوبليكس DA2")
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        type_match = re.search(type_pattern, text)
        if type_match:
            self.data.page4_header_unit_type = type_match.group(1)
            logger.debug(f"Page 4 header unit type: {self.data.page4_header_unit_type}")
    
    def _extract_page5_first_floor(self):
        """Extract first floor area and unit type from page 5 header"""
        text = self.data.raw_text_by_page.get(5, "")
        
        # Extract area from header
        area_pattern = r'(\d+\.?\d*)\s*m[²2]?'
        match = re.search(area_pattern, text)
        if match:
            area_val = float(match.group(1))
            if self.data.first_floor_area == 0.0:
                self.data.first_floor_area = area_val
            # Page 5 header area = First_Floor_Area
            self.data.page5_header_area = area_val
        
        # Extract unit type from page 5 header
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        type_match = re.search(type_pattern, text)
        if type_match:
            self.data.page5_header_unit_type = type_match.group(1)
            logger.debug(f"Page 5 header unit type: {self.data.page5_header_unit_type}")
    
    def _extract_page6_upper_floor(self):
        """Extract upper/second floor area and unit type from page 6 header"""
        text = self.data.raw_text_by_page.get(6, "")
        
        # Extract area from header
        area_pattern = r'(\d+\.?\d*)\s*m[²2]?'
        match = re.search(area_pattern, text)
        if match:
            area_val = float(match.group(1))
            if self.data.upper_floor_area == 0.0:
                self.data.upper_floor_area = area_val
                self.data.second_floor_area = area_val
            # Page 6 header area = Second_Floor_Area
            self.data.page6_header_area = area_val
        
        # Extract unit type from page 6 header
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        type_match = re.search(type_pattern, text)
        if type_match:
            self.data.page6_header_unit_type = type_match.group(1)
            logger.debug(f"Page 6 header unit type: {self.data.page6_header_unit_type}")
    
    def _extract_page7_roof(self):
        """Extract unit type from page 7 header (Roof/السطح)"""
        text = self.data.raw_text_by_page.get(7, "")
        
        # Extract unit type from page 7 header
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        type_match = re.search(type_pattern, text)
        if type_match:
            self.data.page7_header_unit_type = type_match.group(1)
            logger.debug(f"Page 7 header unit type: {self.data.page7_header_unit_type}")
    
    def _extract_page8_facades(self):
        """Extract unit type from page 8 headers (Front & Back facades)"""
        text = self.data.raw_text_by_page.get(8, "")
        
        # Extract unit type(s) from page 8 - may have two sections
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        type_matches = re.findall(type_pattern, text)
        if type_matches:
            # First occurrence = Front facade (الواجهة الأمامية)
            self.data.page8_sec1_unit_type = type_matches[0]
            # Second occurrence (if exists) = Back facade (الواجهة الخلفية)
            if len(type_matches) >= 2:
                self.data.page8_sec2_unit_type = type_matches[1]
            else:
                self.data.page8_sec2_unit_type = type_matches[0]
            logger.debug(f"Page 8 unit types: Front={self.data.page8_sec1_unit_type}, Back={self.data.page8_sec2_unit_type}")
    
    def _extract_page9_side_facade(self):
        """Extract unit type from page 9 header (Side facade/الواجهة الجانبية)"""
        text = self.data.raw_text_by_page.get(9, "")
        
        # Extract unit type from page 9 header
        type_pattern = r'\b(DA\d|DB\d|V\d[A-Z]|C\d+|VS\d|VT\d)\b'
        type_match = re.search(type_pattern, text)
        if type_match:
            self.data.page9_header_unit_type = type_match.group(1)
            logger.debug(f"Page 9 header unit type: {self.data.page9_header_unit_type}")
    
    def _extract_facade_style(self):
        """Detect facade style using Claude Vision as primary method.
        
        Traditional (T): Has half-circle/arch above door (like attached image shows)
        Modern (M): All doors have flat/rectangular tops, no arches
        
        NOTE: Edge detection was removed because it produced false positives.
        The arch detection algorithm was incorrectly finding circles/ellipses in
        architectural drawings that weren't actual door arches.
        """
        logger.info("Extracting facade style...")
        
        # Known facade types by unit type (from ROSHN specifications)
        # STEP 1: Use Claude Vision (primary, most reliable for this task)
        logger.info("Step 1: Using Claude Vision for facade detection...")
        try:
            result = self._extract_facade_with_multiple_pages()
            if result and 'facade_type' in result:
                facade = result['facade_type']
                # Validate result
                if facade in ['T', 'M']:
                    self.data.facade_type = facade
                    logger.info(f"Facade type from Claude Vision: {self.data.facade_type}")
                    return
        except Exception as e:
            logger.error(f"Claude Vision facade extraction failed: {e}")
        
        # STEP 2: Try Gemini Vision as fallback
        logger.info("Step 2: Trying Gemini Vision for facade detection...")
        for page_num in [7, 8]:
            if page_num >= len(self.images):
                continue
            
            try:
                page_img = self.images[page_num]
                result = self._extract_with_gemini_vision(page_img, "facade")
                
                if result and 'facade_type' in result:
                    facade = result['facade_type']
                    if facade in ['T', 'M']:
                        self.data.facade_type = facade
                        logger.info(f"Facade type from Gemini Vision (page {page_num+1}): {self.data.facade_type}")
                        return
                    
            except Exception as e:
                logger.error(f"Gemini Vision facade extraction failed for page {page_num+1}: {e}")
                continue
        
        # Final fallback: Default to Modern (M) since Traditional requires clear arch
        # If no arch is clearly detected by Vision APIs, assume Modern
        self.data.facade_type = "M"
        logger.info(f"Facade type defaulted to M (Modern) - no clear arch detected")
    
    def _count_vertical_lines(self, img: np.ndarray, page_num: int) -> int:
        """Count vertical lines in facade image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        
        vertical_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10 and abs(y2 - y1) > 30:
                    vertical_count += 1
        
        return vertical_count
    
    def _detect_arch_shapes(self, img: np.ndarray, page_num: int) -> bool:
        """Detect arch/half-circle shapes (Traditional style indicator)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Hough Circle Transform to detect circular shapes
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=100)
        
        arch_count = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Check if it's a semi-circle (arch) by checking position in image
                # Arches are typically in the middle-upper portion of doors/windows
                arch_count += 1
        
        # Also detect using edge contours for arch shapes
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 10:
                continue
            
            # Fit ellipse to contour
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (MA, ma), angle = ellipse
                    
                    # Check for arch-like shape (semi-ellipse)
                    aspect_ratio = max(MA, ma) / (min(MA, ma) + 0.001)
                    if 1.5 < aspect_ratio < 3.0 and 20 < min(MA, ma) < 150:
                        arch_count += 1
                except:
                    pass
        
        logger.debug(f"Page {page_num} arch detection: count={arch_count}")
        return arch_count > 3  # Threshold for detecting Traditional style
    
    def _extract_facade_with_multiple_pages(self) -> dict:
        """Extract facade type by analyzing multiple pages together"""
        
        # Collect images from pages 8 and 9 (indices 7 and 8)
        images_content = []
        for page_num in [7, 8]:
            if page_num < len(self.images):
                img_base64 = self._image_to_base64(self.images[page_num])
                images_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64,
                    }
                })
        
        if not images_content:
            return {}
        
        prompt = """Examine these ROSHN building facade elevation images to determine architectural style.

**STEP-BY-STEP ANALYSIS:**

1. Find the FRONT FACADE (الواجهة الأمامية) - the main street-facing elevation
2. Locate the MAIN ENTRANCE DOOR on the ground floor
3. Look DIRECTLY ABOVE the main entrance door
4. Check if there is a CURVED/ARCHED shape above the door (not flat/rectangular)

**WHAT TO LOOK FOR:**
- An ARCH is a semi-circular or curved opening above a door
- It creates a rounded top instead of a flat horizontal line
- May appear as an arched window, transom, or decorative element above the door
- The curve is clear and noticeable, forming part of the door surround

**CLASSIFICATION:**

**TRADITIONAL (T):**
- The main entrance door has a CURVED/ARCHED element above it
- The arch creates a non-rectangular shape above the door
- Even a small curved window or arch detail above the door = Traditional

**MODERN (M):**
- The main entrance door has ONLY flat/rectangular shapes above it
- No curved elements whatsoever above the main door
- Purely geometric, straight lines only

**IMPORTANT:** Focus on the FRONT FACADE MAIN ENTRANCE - this is the key door to check.

Respond with ONLY the JSON classification:
{"facade_type": "T"} if arch/curve found above main door
{"facade_type": "M"} if no arch/curve found (all flat)"""

        try:
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": images_content + [{"type": "text", "text": prompt}]
                    }
                ],
            )
            
            response_text = message.content[0].text.strip()
            logger.debug(f"Claude Vision multi-page facade response: {response_text}")
            
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Multi-page facade Claude Vision API error: {e}")
            return {}
    
    def _detect_vertical_lines(self, img: np.ndarray, page_num: int) -> bool:
        """Detect vertical lines (modern style indicator)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        
        vertical_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10 and abs(y2 - y1) > 30:
                    vertical_count += 1
        
        # Save debug image
        debug_img = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10 and abs(y2 - y1) > 30:
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.debug_subdir, f"vertical_lines_page{page_num}.png"), debug_img)
        
        return vertical_count > 20
    
    def _detect_red_highlights(self):
        """Detect red highlights on page 2"""
        if len(self.cv_images) < 2:
            return
        
        img = self.cv_images[1]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Red color ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Pink/light red
        lower_pink = np.array([0, 50, 200])
        upper_pink = np.array([20, 150, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        combined_mask = cv2.bitwise_or(red_mask, pink_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_height = img.shape[0]
        top_red = sum(cv2.contourArea(c) for c in contours 
                      if cv2.contourArea(c) > 500 and 
                      cv2.moments(c)["m00"] > 0 and 
                      cv2.moments(c)["m01"] / cv2.moments(c)["m00"] < img_height / 2)
        
        bottom_red = sum(cv2.contourArea(c) for c in contours 
                        if cv2.contourArea(c) > 500 and 
                        cv2.moments(c)["m00"] > 0 and 
                        cv2.moments(c)["m01"] / cv2.moments(c)["m00"] >= img_height / 2)
        
        self.data.red_highlight_neighborhood = top_red > 1000
        self.data.red_highlight_plot = bottom_red > 1000
        
        logger.info(f"Red highlights - Neighborhood: {self.data.red_highlight_neighborhood}, Plot: {self.data.red_highlight_plot}")
        
        # Save debug
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.debug_subdir, "red_highlights.png"), debug_img)
    
    def _post_process(self):
        """Final post-processing"""
        # Ensure all values have defaults if still missing
        if self.data.unit_type in self.UNIT_DIMENSIONS:
            defaults = self.UNIT_DIMENSIONS[self.data.unit_type]
            
            if self.data.land_length == 0.0:
                self.data.land_length = defaults[0]
                self.data.extraction_notes.append(f"Land length defaulted to {defaults[0]}")
            
            if self.data.land_width == 0.0:
                self.data.land_width = defaults[1]
                self.data.extraction_notes.append(f"Land width defaulted to {defaults[1]}")
            
            if self.data.street_width == 0:
                self.data.street_width = defaults[2]
                self.data.extraction_notes.append(f"Street width defaulted to {defaults[2]}")
        
        logger.info(f"Final dimensions: length={self.data.land_length}, width={self.data.land_width}, street={self.data.street_width}")


def extract_pdf_data(pdf_path: str, debug_dir: str = None) -> ExtractedData:
    """Convenience function to extract all data from a PDF"""
    extractor = PDFExtractor(pdf_path, debug_dir)
    return extractor.extract_all()


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/DM1-D02-2A-25-453-01.pdf"
    
    data = extract_pdf_data(pdf_path)
    
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    print(f"Unit Code: {data.unit_code}")
    print(f"Plot Number: {data.plot_number}")
    print(f"Unit Type: {data.unit_type}")
    print(f"\nAreas:")
    print(f"  Land: {data.land_area} m²")
    print(f"  Ground Floor: {data.ground_floor_area} m²")
    print(f"  First Floor: {data.first_floor_area} m²")
    print(f"  Upper Floor: {data.upper_floor_area} m²")
    print(f"  Total GFA: {data.total_floor_area} m²")
    print(f"\nDimensions:")
    print(f"  Length: {data.land_length} m")
    print(f"  Width: {data.land_width} m")
    print(f"  Street: {data.street_width} m")
    print(f"\nFacade: {data.facade_type}")
    print(f"Red Highlights: Neighborhood={data.red_highlight_neighborhood}, Plot={data.red_highlight_plot}")
    if data.extraction_notes:
        print(f"\nNotes: {data.extraction_notes}")
