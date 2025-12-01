# PDF Validation System for ROSHN Property Documents

## Overview

This system validates ROSHN property PDF documents against ground truth data stored in Excel. It performs comprehensive validation including:

- **Page 1**: File name / Unit code verification
- **Page 2**: Reference map, region, plot number, red highlights for neighborhood and plot
- **Page 3**: Site plan with areas table, dimensions, street width, villa type
- **Pages 4-6**: Floor plans with area verification
- **Pages 8-9**: Facade style detection (Modern vs Traditional based on vertical lines near windows)

## Files

| File | Description |
|------|-------------|
| `gui.py` | **Professional GUI Application** with ROSHN branding |
| `main.py` | Command-line entry point - orchestrates the validation process |
| `config.py` | Configuration with Arabic translations, field mappings, validation rules |
| `pdf_extractor.py` | Extracts data from PDFs (text, images, red highlights, dimensions) |
| `excel_loader.py` | Loads ground truth data from Excel |
| `validator.py` | Compares PDF data against ground truth |
| `report_generator.py` | Generates Excel validation report |

## Installation

```bash
# Install required Python packages
pip install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-ara

# For Windows: Ensure poppler is in PATH or in ./poppler folder
```

## Usage

### 🖥️ GUI Application (Recommended)

Launch the professional graphical interface:

```bash
python gui.py
```

**Features:**
- 📁 Browse for PDF files or folders
- 📊 Select ground truth Excel file
- 🤖 Toggle AI analysis on/off with cost estimation
- 📈 Real-time progress monitoring
- 📋 Validation results summary
- 💰 AI cost tracking per session

### Command Line Usage

#### Simple Usage (Default Paths)
```bash
python main.py
```
This uses default paths:
- PDF Directory: `./input_data`
- Ground Truth: `./input_data/ground_truth.xlsx`
- Output: `./output`

### Custom Paths
```bash
python main.py --pdf-dir /path/to/pdf_folder --excel /path/to/ground_truth.xlsx
```

### Single PDF Validation
```bash
python main.py --pdf /path/to/document.pdf --excel /path/to/ground_truth.xlsx
```

### Options
```
--pdf         Path to single PDF file
--pdf-dir     Path to directory containing PDFs (default: ./input_data)
--excel       Path to ground truth Excel file (default: ./input_data/ground_truth.xlsx)
--output      Output directory for reports (default: ./output)
--debug       Enable debug mode (saves intermediate images)
--verbose     Enable verbose logging
```

## Output Report

The system generates an Excel report (`validation_TIMESTAMP.xlsx`) with:
- **Summary**: Overall statistics, per-PDF status
- **Page-by-Page Validation**: Detailed validation organized by PDF page
- **Detailed Results**: All validation checks with PDF vs Ground Truth values
- **Failures Only**: Only mismatches and warnings
- **PDF vs Ground Truth**: Side-by-side comparison

## Validation Checks

### Critical (Must Match)
- Unit code
- Plot number
- Unit type (DA2, V1A, etc.)

### High Priority
- Land area
- Ground floor area
- First floor area
- Upper floor area
- Total floor area (GFA)
- Ground floor annex
- Red highlight presence (neighborhood & plot)

### Medium Priority
- Land dimensions (length, width)
- Street width
- Facade type (M=Modern, T=Traditional)

## Field Mappings (PDF Arabic to Excel)

| PDF Field (Arabic) | Excel Column |
|-------------------|--------------|
| رقم الوحدة السكنية | Unit_Code |
| القطعة رقم | AMANA_Plot_No |
| مساحة الأرض | Land_Area |
| مساحة الطابق الأرضي | Ground_Floor_Area |
| مساحة الطابق الأول | First_Floor_Area |
| مساحة الطابق العلوي | Second_Floor_Area |
| إجمالي مساحة الطوابق | GFA |

## Facade Style Detection

The system detects facade style based on architectural features:

- **Modern (M)**: Has vertical lines/slats near windows on elevation pages
- **Traditional (T)**: Standard window designs without vertical slats

Detection algorithm analyzes pages 8-9 for vertical line patterns using edge detection.

## Debugging

With `--debug` flag, the system saves:
- Converted PDF page images
- Extracted text per page
- Red highlight detection masks
- Vertical line detection results

Debug files are saved to `debug_images/` directory.

## Extending the System

### Adding New Validation Fields

1. Add field to `ExtractedData` dataclass in `pdf_extractor.py`
2. Add extraction logic in appropriate `_extract_pageX_` method
3. Add mapping in `config.py` (`PDF_TO_EXCEL_MAPPING`)
4. Add field to `GroundTruthRecord` in `excel_loader.py`
5. Add validation check in `validator.py`

### Customizing Tolerances

Edit `TOLERANCES` in `config.py`:
```python
TOLERANCES = {
    "area": 0.01,      # Square meters tolerance
    "dimension": 0.01,  # Meters tolerance
    "percentage": 0.01, # 1% tolerance
}
```

## Troubleshooting

### Common Issues

1. **OCR not working**: Install Tesseract with Arabic support
   ```bash
   sudo apt-get install tesseract-ocr tesseract-ocr-ara
   ```

2. **PDF conversion fails**: Install poppler-utils
   ```bash
   sudo apt-get install poppler-utils
   ```

3. **Areas not extracted**: Check that the PDF has selectable text (not just images)

4. **Red highlights not detected**: Adjust HSV color ranges in `_detect_red_highlights()`

## License

Internal use only - ROSHN Group
