"""
Configuration file for PDF validation system.
Contains Arabic translations, field mappings, and validation rules.
"""

# Arabic to English translations for PDF fields
ARABIC_TRANSLATIONS = {
    # Page 1
    "رقم الوحدة السكنية": "Housing Unit Number",
    
    # Page 2 - Headers
    "الخريطة المرجعية": "Reference Map",
    "المنطقة": "Region",
    "القطعة رقم": "Plot Number",
    
    # Page 3 - Site plan
    "الموقع العام": "General Location",
    "جدول المساحات": "Area Table",
    "مساحة الأرض": "Land Area",
    "مساحة الطابق الأرضي": "Ground Floor Area",
    "مساحة الملحق الأرضي": "Ground Floor Annex",
    "إجمالي مساحة الطوابق": "Total Floor Area",
    "مساحة الطابق الأول": "First Floor Area",
    "مساحة الطابق العلوي": "Upper Floor Area",
    "شارع عرض": "Street Width",
    "حديقة": "Garden",
    "متر": "meter",
    
    # Pages 4-6 - Floor Plans
    "النموذج": "Model",
    "المساحة": "Area",
    "الطابق الأرضي": "Ground Floor",
    "الطابق الأول": "First Floor",
    "الملحق العلوي": "Upper Annex",
    "السطح": "Roof",
    
    # Rooms
    "الفناء الخارجي": "External Courtyard",
    "غرفة المعيشة": "Living Room",
    "مطبخ": "Kitchen",
    "غرفة الطعام": "Dining Room",
    "مخزن": "Storage",
    "ممر": "Corridor",
    "دورة مياه": "Bathroom",
    "مجلس": "Majlis",
    "غرفة السائق": "Driver's Room",
    "موقف السيارات": "Car Parking",
    "خزانة خدمات": "Utility Closet",
    "المدخل الرئيسي": "Main Entrance",
    "جار ملاصق": "Adjacent Neighbor",
    "غرفة نوم": "Bedroom",
    "مكتب": "Office",
    "غرفة النوم الرئيسية": "Master Bedroom",
    "خزانة الملابس": "Wardrobe",
    "درج": "Stairs",
    "غرفة الخادمة": "Maid's Room",
    "غرفة الغسيل": "Laundry Room",
    "فناء علوي": "Upper Courtyard",
    
    # Pages 8-9 - Elevations
    "الواجهة الأمامية": "Front Facade",
    "الواجهة الخلفية": "Back Facade",
    "الواجهة الجانبية": "Side Facade",
}

# Mapping from PDF fields to Excel columns
PDF_TO_EXCEL_MAPPING = {
    "unit_code": "Unit_Code",
    "plot_number": "AMANA_Plot_No",
    "neighborhood": "NBHD_Name",
    "reference_map": None,
    "region": None,
    
    # Areas
    "land_area": "Land_Area",
    "ground_floor_area": "Ground_Floor_Area",
    "ground_floor_annex": "Ground_Floor_Annex_Driver",
    "total_floor_area": "GFA",
    "first_floor_area": "First_Floor_Area",
    "upper_floor_area": "Second_Floor_Area",
    
    # Dimensions
    "land_length": "Land_Length",
    "land_width": "Land_Width",
    "street_width": "Street_Width",
    
    # Model/Type
    "unit_type": "Unit_Type",
    "facade_type": "Façade_Type",
    
    # Additional fields
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms/powderroom",
    "maids_room": "Maids_Room",
    "drivers_room": "Drivers_Room",
    "terrace": "Terrace",
    "terrace_area": "Terrace_Area",
}

# Unit type codes
UNIT_TYPE_MAPPING = {
    "DA2": "DA2",
    "DA3": "DA3",
    "V1A": "V1A",
    "V2B": "V2B",
    "V2C": "V2C",
    "VT1": "VT1",
    "VS1": "VS1",
    "C20": "C20",
    "DB2": "DB2",
    "DB3": "DB3",
}

# Facade type mapping (Modern = M, Traditional = T)
FACADE_STYLE_MAPPING = {
    "modern": "M",
    "traditional": "T",
}

# Tolerance levels for numeric comparisons
TOLERANCES = {
    "area": 0.01,
    "dimension": 0.01,
    "percentage": 0.01,
}

# Validation severity levels
SEVERITY = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
}

# Field validation rules
VALIDATION_RULES = {
    "unit_code": {"severity": SEVERITY["CRITICAL"], "validation_type": "exact_match"},
    "plot_number": {"severity": SEVERITY["CRITICAL"], "validation_type": "exact_match"},
    "unit_type": {"severity": SEVERITY["CRITICAL"], "validation_type": "exact_match"},
    "land_area": {"severity": SEVERITY["HIGH"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "ground_floor_area": {"severity": SEVERITY["HIGH"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "first_floor_area": {"severity": SEVERITY["HIGH"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "upper_floor_area": {"severity": SEVERITY["HIGH"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "total_floor_area": {"severity": SEVERITY["HIGH"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "land_length": {"severity": SEVERITY["MEDIUM"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "land_width": {"severity": SEVERITY["MEDIUM"], "validation_type": "numeric_tolerance", "tolerance": 0.01},
    "street_width": {"severity": SEVERITY["MEDIUM"], "validation_type": "exact_match"},
    "facade_type": {"severity": SEVERITY["MEDIUM"], "validation_type": "mapped_match"},
    "red_highlight_neighborhood": {"severity": SEVERITY["HIGH"], "validation_type": "presence_check"},
    "red_highlight_plot": {"severity": SEVERITY["HIGH"], "validation_type": "presence_check"},
}

# Debug settings
DEBUG_CONFIG = {
    "save_intermediate_images": True,
    "verbose_logging": True,
    "save_ocr_text": True,
    "image_output_dir": "/home/claude/pdf_validator/debug_images",
}

# Vision Model Configuration
# Priority order: 1=normal text extraction, 2=Claude Vision, 3=Gemini Vision
VISION_MODEL_CONFIG = {
    "default_model": "claude",  # Options: "claude", "gemini", "auto"
    "fallback_enabled": True,
    "models": {
        "claude": {
            "name": "Claude Vision (claude-sonnet-4-20250514)",
            "priority": 1,
            "enabled": True,
        },
        "gemini": {
            "name": "Gemini Vision (gemini-2.0-flash)",
            "priority": 2,
            "enabled": True,
        },
    },
    # Fields that require vision model (normal text extraction unreliable)
    "vision_required_fields": [
        "dimensions",  # Page 3 plot dimensions
        "facade",      # Page 8 facade style detection
    ],
    # Fields that can use text extraction first
    "text_extraction_fields": [
        "unit_code",           # Page 1
        "page2_data",          # Page 2 (NBHD, Unit Type, Plot No from text)
        "page3_areas",         # Page 3 footer areas table
        "page4_header",        # Page 4 header
        "page5_header",        # Page 5 header
        "page6_header",        # Page 6 header
        "page7_header",        # Page 7 header
        "page8_header",        # Page 8 headers
        "page9_header",        # Page 9 header
    ],
}
