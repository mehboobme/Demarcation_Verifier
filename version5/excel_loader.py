"""
Excel Ground Truth Loader Module
Loads and processes the ground truth data from Excel for validation.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class GroundTruthRecord:
    """Container for a single ground truth record"""
    # Identifiers
    parcel_number: int = 0
    district: str = ""
    region_site: str = ""
    unit_code: str = ""
    amana_block_no: int = 0
    amana_plot_no: int = 0
    nbhd_name: str = ""
    phase: int = 0
    
    # Property type
    main_land_use: str = ""
    unit_type: str = ""
    sub_land_use: str = ""
    attached_standalone: str = ""
    
    # Areas
    land_area: float = 0.0
    plot_extension: float = 0.0
    far: float = 0.0
    gfa: float = 0.0
    bua: float = 0.0
    
    # Dimensions
    land_shape: str = ""
    land_length: float = 0.0
    land_width: float = 0.0
    
    # Location attributes
    corner_unit: str = ""
    cul_de_sac: str = ""
    green_spine_view: str = ""
    car_parking: int = 0
    orientation: str = ""
    front: str = ""
    rear: str = ""
    side: str = ""
    number_of_streets: int = 0
    street_width: int = 0
    no_of_facades: int = 0
    
    # Facade
    facade_type: str = ""
    facade_color: int = 0
    
    # Building specs
    storeys: str = ""
    height: float = 0.0
    ground_floor_area: float = 0.0
    ground_floor_annex: float = 0.0
    first_floor_area: float = 0.0
    second_floor_area: float = 0.0
    
    # Balcony/Terrace
    balcony: str = ""
    balcony_area: float = 0.0
    terrace: str = ""
    terrace_area: float = 0.0
    
    # Rooms
    bedrooms: int = 0
    bathrooms: int = 0
    living_room_majlis: int = 0
    multi_purpose_room: int = 0
    maids_room: str = ""
    drivers_room: str = ""
    
    # Raw data reference
    raw_data: Dict = None


class GroundTruthLoader:
    """Loads and manages ground truth data from Excel"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df: pd.DataFrame = None
        self.records: Dict[int, GroundTruthRecord] = {}  # Keyed by plot number
        
    def load(self) -> Dict[int, GroundTruthRecord]:
        """Load all records from Excel"""
        logger.info(f"Loading ground truth from: {self.excel_path}")
        
        self.df = pd.read_excel(self.excel_path)
        logger.info(f"Loaded {len(self.df)} records")
        
        # Process each row
        for idx, row in self.df.iterrows():
            record = self._row_to_record(row)
            if record.amana_plot_no:
                self.records[record.amana_plot_no] = record
        
        logger.info(f"Processed {len(self.records)} valid records")
        return self.records
    
    def _row_to_record(self, row: pd.Series) -> GroundTruthRecord:
        """Convert a DataFrame row to a GroundTruthRecord"""
        record = GroundTruthRecord()
        
        # Helper function to safely get values
        def safe_get(col, default=None):
            try:
                val = row.get(col)
                if pd.isna(val):
                    return default
                return val
            except:
                return default
        
        def safe_int(col, default=0):
            val = safe_get(col)
            if val is None:
                return default
            try:
                return int(val)
            except:
                return default
        
        def safe_float(col, default=0.0):
            val = safe_get(col)
            if val is None:
                return default
            try:
                return float(val)
            except:
                return default
        
        def safe_str(col, default=""):
            val = safe_get(col)
            if val is None:
                return default
            return str(val).strip()
        
        # Identifiers
        record.parcel_number = safe_int('Parcel Number on CAD (do not delete) ')
        record.district = safe_str('District')
        record.region_site = safe_str('Region_Site')
        record.unit_code = safe_str('Unit_Code')
        record.amana_block_no = safe_int('AMANA_Block_No')
        record.amana_plot_no = safe_int('AMANA_Plot_No')
        record.nbhd_name = safe_str('NBHD_Name')
        record.phase = safe_int('Phase')
        
        # Property type
        record.main_land_use = safe_str('Main_Land_Use')
        record.unit_type = safe_str('Unit_Type')
        record.sub_land_use = safe_str('Sub_Land_Use')
        record.attached_standalone = safe_str('Attached_StandAlone')
        
        # Areas
        record.land_area = safe_float('Land_Area')
        record.plot_extension = safe_float('Plot_Extension')
        record.far = safe_float('FAR')
        record.gfa = safe_float('GFA')
        record.bua = safe_float('BUA')
        
        # Dimensions
        record.land_shape = safe_str('Land_Shape')
        record.land_length = safe_float('Land_Length')
        record.land_width = safe_float('Land_Width')
        
        # Location attributes
        record.corner_unit = safe_str('Corner_Unit')
        record.cul_de_sac = safe_str('Cul_De_Sac')
        record.green_spine_view = safe_str('Green_Spine_View')
        record.car_parking = safe_int('Car_Parking')
        record.orientation = safe_str('Orientation')
        record.front = safe_str('Front')
        record.rear = safe_str('Rear')
        record.side = safe_str('Side')
        record.number_of_streets = safe_int('Number_Of_Streets')
        record.street_width = safe_int('Street_Width')
        record.no_of_facades = safe_int('No_Of_Facades')
        
        # Facade
        record.facade_type = safe_str('Façade_Type')
        record.facade_color = safe_int('Façade_Color')
        
        # Building specs
        record.storeys = safe_str('Storeys')
        record.height = safe_float('Height')
        record.ground_floor_area = safe_float('Ground_Floor_Area')
        record.ground_floor_annex = safe_float('Ground_Floor_Annex_Driver')
        record.first_floor_area = safe_float('First_Floor_Area')
        record.second_floor_area = safe_float('Second_Floor_Area')
        
        # Balcony/Terrace
        record.balcony = safe_str('Balcony')
        record.balcony_area = safe_float('Balcony_Area')
        record.terrace = safe_str('Terrace')
        record.terrace_area = safe_float('Terrace_Area')
        
        # Rooms
        record.bedrooms = safe_int('Bedrooms')
        record.bathrooms = safe_int('Bathrooms/powderroom')
        record.living_room_majlis = safe_int('Living_Room_Majlis ')
        record.multi_purpose_room = safe_int('Multi_Purpose_Room')
        record.maids_room = safe_str('Maids_Room')
        record.drivers_room = safe_str('Drivers_Room')
        
        # Keep raw data
        record.raw_data = row.to_dict()
        
        return record
    
    def get_record_by_plot(self, plot_number: int) -> Optional[GroundTruthRecord]:
        """Get a record by plot number"""
        return self.records.get(plot_number)
    
    def get_record_by_unit_code(self, unit_code: str) -> Optional[GroundTruthRecord]:
        """Get a record by unit code"""
        for record in self.records.values():
            if record.unit_code == unit_code:
                return record
        return None
    
    def get_all_plot_numbers(self) -> List[int]:
        """Get list of all plot numbers"""
        return list(self.records.keys())
    
    def get_all_unit_codes(self) -> List[str]:
        """Get list of all unit codes"""
        return [r.unit_code for r in self.records.values()]


def load_ground_truth(excel_path: str) -> Dict[int, GroundTruthRecord]:
    """Convenience function to load ground truth data"""
    loader = GroundTruthLoader(excel_path)
    return loader.load()


if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    else:
        excel_path = "/mnt/user-data/uploads/ground_truth.xlsx"
    
    loader = GroundTruthLoader(excel_path)
    records = loader.load()
    
    print("\n" + "="*60)
    print("GROUND TRUTH DATA:")
    print("="*60)
    
    for plot_no, record in records.items():
        print(f"\nPlot {plot_no}: {record.unit_code}")
        print(f"  Type: {record.unit_type}")
        print(f"  Land Area: {record.land_area} m²")
        print(f"  GFA: {record.gfa} m²")
        print(f"  Ground Floor: {record.ground_floor_area} m²")
        print(f"  First Floor: {record.first_floor_area} m²")
        print(f"  Second Floor: {record.second_floor_area} m²")
        print(f"  Dimensions: {record.land_length} x {record.land_width} m")
        print(f"  Street Width: {record.street_width} m")
        print(f"  Facade Type: {record.facade_type}")
