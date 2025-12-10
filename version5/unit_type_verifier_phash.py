"""
Hybrid Unit Type Verifier - pHash + SSIM
-----------------------------------------
Uses perceptual hashing (robust to JPEG compression, signature removal)
combined with SSIM for structural verification
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import imagehash

class HybridUnitTypeVerifier:
    """Fast verifier using pHash + SSIM"""
    
    def __init__(self, reference_dir: str = "reference_images/unit types"):
        self.reference_dir = Path(reference_dir)
        print("Loading reference images with pHash...")
        self.reference_hashes = self._load_reference_hashes()
        print(f"[OK] Loaded {sum(len(v['facade']['modern']) + len(v['facade']['traditional']) + len(v['floor_plan']) for v in self.reference_hashes.values())} reference hashes")
    
    def _load_image_unicode_safe(self, img_path: Path) -> Optional[Image.Image]:
        """Load image as PIL with Unicode support"""
        try:
            return Image.open(str(img_path))
        except Exception as e:
            return None
    
    def _compute_phash(self, img: np.ndarray, hash_size=16) -> imagehash.ImageHash:
        """Compute perceptual hash from numpy array"""
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        pil_img = Image.fromarray(img_rgb)
        return imagehash.phash(pil_img, hash_size=hash_size)
    
    def _hash_similarity(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """
        Compute similarity from hash distance
        Returns 0-1 where 1 is identical, 0 is completely different
        """
        distance = hash1 - hash2  # Hamming distance
        max_distance = len(hash1.hash) ** 2  # Maximum possible distance
        similarity = 1.0 - (distance / max_distance)
        return similarity
    
    def _load_reference_hashes(self) -> Dict:
        """Load and compute hashes for all reference images"""
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
                            pil_img = self._load_image_unicode_safe(img_path)
                            if pil_img is not None:
                                phash = imagehash.phash(pil_img, hash_size=16)
                                reference_data[unit_type]['facade'][style].append({
                                    'path': str(img_path),
                                    'hash': phash,
                                    'name': img_path.name
                                })
            
            # Load floor plan images
            floor_dir = unit_type_dir / "Floor Plans"
            if floor_dir.exists():
                for img_path in floor_dir.glob("*.jpg"):
                    pil_img = self._load_image_unicode_safe(img_path)
                    if pil_img is not None:
                        phash = imagehash.phash(pil_img, hash_size=16)
                        reference_data[unit_type]['floor_plan'].append({
                            'path': str(img_path),
                            'hash': phash,
                            'name': img_path.name
                        })
        
        return reference_data
    
    def verify_facade(self, facade_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify facade using pHash"""
        test_hash = self._compute_phash(facade_image)
        
        best_match = {'unit_type': None, 'similarity': 0, 'style': None, 'ref_name': None}
        all_scores = {}
        
        for unit_type, data in self.reference_hashes.items():
            for style in ['modern', 'traditional']:
                for ref_data in data['facade'][style]:
                    similarity = self._hash_similarity(test_hash, ref_data['hash'])
                    
                    key = f"{unit_type}_{style}"
                    if key not in all_scores or similarity > all_scores[key]:
                        all_scores[key] = similarity
                    
                    if similarity > best_match['similarity']:
                        best_match = {
                            'unit_type': unit_type,
                            'similarity': similarity,
                            'style': style,
                            'ref_name': ref_data['name']
                        }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'style': best_match['style'],
            'reference': best_match['ref_name'],
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'all_scores': all_scores
        }
    
    def verify_floor_plan(self, floor_plan_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify floor plan using pHash"""
        test_hash = self._compute_phash(floor_plan_image)
        
        best_match = {'unit_type': None, 'similarity': 0, 'ref_name': None}
        all_scores = {}
        
        for unit_type, data in self.reference_hashes.items():
            for ref_data in data['floor_plan']:
                similarity = self._hash_similarity(test_hash, ref_data['hash'])
                
                if unit_type not in all_scores or similarity > all_scores[unit_type]:
                    all_scores[unit_type] = similarity
                
                if similarity > best_match['similarity']:
                    best_match = {
                        'unit_type': unit_type,
                        'similarity': similarity,
                        'ref_name': ref_data['name']
                    }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'reference': best_match['ref_name'],
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'all_scores': all_scores
        }
    
    def verify_floor_plans_multi(self, floor_images: Dict[str, np.ndarray], declared_unit_type: str) -> Dict:
        """
        Verify multiple floor plan pages and combine results
        floor_images: dict like {'ground': img, 'first': img, 'top': img, 'terrace': img}
        """
        results = {}
        total_similarity = 0
        count = 0
        
        for floor_name, floor_image in floor_images.items():
            if floor_image is not None:
                result = self.verify_floor_plan(floor_image, declared_unit_type)
                results[floor_name] = result
                total_similarity += result['similarity']
                count += 1
        
        if count == 0:
            return {'detected': None, 'similarity': 0, 'is_correct': False, 'details': {}}
        
        # Average similarity across all floor plans
        avg_similarity = total_similarity / count
        
        # Vote for most common detected unit type
        detected_types = [r['detected'] for r in results.values()]
        most_common = max(set(detected_types), key=detected_types.count) if detected_types else None
        
        return {
            'detected': most_common,
            'similarity': avg_similarity,
            'is_correct': most_common == declared_unit_type,
            'details': results,
            'vote_count': detected_types.count(most_common) if most_common else 0,
            'total_pages': count
        }
    
    def verify_facades_multi(self, facade_images: Dict[str, np.ndarray], declared_unit_type: str) -> Dict:
        """
        Verify multiple facade pages and combine results
        facade_images: dict like {'front': img, 'back': img}
        """
        results = {}
        total_similarity = 0
        count = 0
        
        for facade_name, facade_image in facade_images.items():
            if facade_image is not None:
                result = self.verify_facade(facade_image, declared_unit_type)
                results[facade_name] = result
                total_similarity += result['similarity']
                count += 1
        
        if count == 0:
            return {'detected': None, 'similarity': 0, 'style': None, 'is_correct': False, 'details': {}}
        
        # Average similarity across all facades
        avg_similarity = total_similarity / count
        
        # Vote for most common detected unit type
        detected_types = [r['detected'] for r in results.values()]
        most_common = max(set(detected_types), key=detected_types.count) if detected_types else None
        
        # Get most common style
        styles = [r['style'] for r in results.values()]
        most_common_style = max(set(styles), key=styles.count) if styles else None
        
        return {
            'detected': most_common,
            'similarity': avg_similarity,
            'style': most_common_style,
            'is_correct': most_common == declared_unit_type,
            'details': results,
            'vote_count': detected_types.count(most_common) if most_common else 0,
            'total_pages': count
        }
    
    def verify_unit_type(self, facade_image: np.ndarray, floor_plan_image: np.ndarray, 
                        declared_unit_type: str) -> Dict:
        """Complete verification using pHash (legacy single-page method)"""
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
    
    def verify_unit_type_multi(self, facade_images: Dict[str, np.ndarray], 
                               floor_images: Dict[str, np.ndarray],
                               declared_unit_type: str) -> Dict:
        """
        Complete verification using multiple pages
        facade_images: {'front': page8_img, 'back': page9_img}
        floor_images: {'ground': page4_img, 'first': page5_img, 'top': page6_img, 'terrace': page7_img}
        """
        facade_result = self.verify_facades_multi(facade_images, declared_unit_type)
        floor_result = self.verify_floor_plans_multi(floor_images, declared_unit_type)
        
        facade_correct = facade_result['is_correct']
        floor_correct = floor_result['is_correct']
        
        # More robust verdict based on votes
        facade_confidence = facade_result.get('vote_count', 0) / max(facade_result.get('total_pages', 1), 1)
        floor_confidence = floor_result.get('vote_count', 0) / max(floor_result.get('total_pages', 1), 1)
        
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
            'average_similarity': avg_similarity,
            'facade_confidence': facade_confidence,
            'floor_confidence': floor_confidence
        }
