"""
Fast Unit Type Verifier using Structural Similarity
-----------------------------------------------------
Since reference images are extracted from actual PDFs, use SSIM for fast exact matching
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class FastUnitTypeVerifier:
    """Fast verifier using SSIM for structural matching"""
    
    def __init__(self, reference_dir: str = "reference_images/unit types"):
        self.reference_dir = Path(reference_dir)
        print("Loading reference images for fast SSIM matching...")
        self.reference_images = self._load_reference_images()
        print(f"[OK] Loaded {sum(len(v['facade']['modern']) + len(v['facade']['traditional']) + len(v['floor_plan']) for v in self.reference_images.values())} reference images")
    
    def _load_image_unicode_safe(self, img_path: Path) -> Optional[np.ndarray]:
        """Load image with Unicode path support"""
        try:
            pil_img = Image.open(str(img_path))
            img_array = np.array(pil_img)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            return img_bgr
        except Exception as e:
            return None
    
    def _preprocess_for_ssim(self, img: np.ndarray, target_size=(512, 512)) -> np.ndarray:
        """Preprocess image for SSIM comparison"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Resize to standard size
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute structural similarity between two images"""
        # Preprocess both images
        proc1 = self._preprocess_for_ssim(img1)
        proc2 = self._preprocess_for_ssim(img2)
        
        # Compute SSIM with data_range specified for float images
        score, _ = ssim(proc1, proc2, full=True, data_range=1.0)
        return score
    
    def _load_reference_images(self) -> Dict:
        """Load all reference images"""
        reference_data = {}
        
        for unit_type_dir in sorted(self.reference_dir.iterdir()):
            if not unit_type_dir.is_dir():
                continue
            
            unit_type = unit_type_dir.name
            reference_data[unit_type] = {
                'facade': {'modern': [], 'traditional': []},
                'floor_plan': []
            }
            
            # Load facade images
            facade_dir = unit_type_dir / "Facade"
            if facade_dir.exists():
                for style in ['modern', 'traditional']:
                    style_dir = facade_dir / style
                    if style_dir.exists():
                        for img_path in style_dir.glob("*.jpg"):
                            img = self._load_image_unicode_safe(img_path)
                            if img is not None:
                                reference_data[unit_type]['facade'][style].append({
                                    'path': str(img_path),
                                    'image': img
                                })
            
            # Load floor plan images
            floor_dir = unit_type_dir / "Floor Plans"
            if floor_dir.exists():
                for img_path in floor_dir.glob("*.jpg"):
                    img = self._load_image_unicode_safe(img_path)
                    if img is not None:
                        reference_data[unit_type]['floor_plan'].append({
                            'path': str(img_path),
                            'image': img
                        })
        
        return reference_data
    
    def verify_facade(self, facade_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify facade using SSIM"""
        best_match = {'unit_type': None, 'similarity': 0, 'style': None}
        
        for unit_type, data in self.reference_images.items():
            for style in ['modern', 'traditional']:
                for ref_data in data['facade'][style]:
                    similarity = self._compute_ssim(facade_image, ref_data['image'])
                    
                    if similarity > best_match['similarity']:
                        best_match = {
                            'unit_type': unit_type,
                            'similarity': similarity,
                            'style': style
                        }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'style': best_match['style'],
            'is_correct': best_match['unit_type'] == declared_unit_type
        }
    
    def verify_floor_plan(self, floor_plan_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify floor plan using SSIM"""
        best_match = {'unit_type': None, 'similarity': 0}
        
        for unit_type, data in self.reference_images.items():
            for ref_data in data['floor_plan']:
                similarity = self._compute_ssim(floor_plan_image, ref_data['image'])
                
                if similarity > best_match['similarity']:
                    best_match = {
                        'unit_type': unit_type,
                        'similarity': similarity
                    }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'is_correct': best_match['unit_type'] == declared_unit_type
        }
    
    def verify_unit_type(self, facade_image: np.ndarray, floor_plan_image: np.ndarray, 
                        declared_unit_type: str) -> Dict:
        """Complete verification using SSIM"""
        facade_result = self.verify_facade(facade_image, declared_unit_type)
        floor_result = self.verify_floor_plan(floor_plan_image, declared_unit_type)
        
        facade_correct = facade_result['is_correct']
        floor_correct = floor_result['is_correct']
        
        if facade_correct and floor_correct:
            verdict = "PASS"
        elif facade_correct or floor_correct:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"
        
        avg_similarity = (facade_result['similarity'] + floor_result['similarity']) / 2
        
        return {
            'facade_verification': facade_result,
            'floor_plan_verification': floor_result,
            'overall_correct': facade_correct and floor_correct,
            'verdict': verdict,
            'average_similarity': avg_similarity
        }
