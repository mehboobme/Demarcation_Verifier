"""
GPU-Accelerated Unit Type Verifier with Blue Boundary Cropping
----------------------------------------------------------------
Uses CUDA-accelerated feature matching (ORB/SIFT) + pHash
Crops floor plans to blue dotted boundary region for better matching
Supports mirror image matching for flipped floor plans
Enhanced with data augmentation and 60% similarity threshold
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import imagehash
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class GPUUnitTypeVerifier:
    """GPU-accelerated verifier with blue boundary cropping and mirror image support"""
    
    # Minimum similarity threshold (60%)
    SIMILARITY_THRESHOLD = 0.60
    
    def __init__(self, reference_dir: str = "reference_images/unit types", use_gpu: bool = True):
        self.reference_dir = Path(reference_dir)
        self.use_gpu = use_gpu
        
        # Check CUDA availability
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0 if use_gpu else False
        if self.use_gpu and not self.cuda_available:
            logger.warning("CUDA not available, falling back to CPU")
            self.use_gpu = False
        
        logger.info(f"GPU Mode: {'ENABLED (CUDA)' if self.cuda_available else 'DISABLED (CPU)'}")
        
        # Initialize feature detector
        if self.cuda_available:
            # Use CUDA ORB for GPU
            self.orb = cv2.cuda.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
            logger.info("Using CUDA-accelerated ORB detector")
        else:
            # Fallback to CPU ORB
            self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
            logger.info("Using CPU ORB detector")
        
        # BFMatcher for feature matching
        if self.cuda_available:
            self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        print("Loading reference images with GPU-accelerated features...")
        self.reference_data = self._load_reference_features()
        
        # Count original and augmented references
        total_refs = 0
        augmented_refs = 0
        for v in self.reference_data.values():
            for ref in v['floor_plan']:
                total_refs += 1
                if ref.get('augmented'):
                    augmented_refs += 1
            total_refs += len(v['facade']['modern']) + len(v['facade']['traditional'])
        
        print(f"[OK] Loaded {total_refs} reference images ({augmented_refs} augmented) with features")
    
    def _apply_augmentation(self, image: np.ndarray, mirror: bool = False) -> np.ndarray:
        """
        Apply data augmentation to improve matching robustness
        
        Args:
            image: Input image
            mirror: If True, horizontally flip the image
        
        Returns:
            Augmented image
        """
        if mirror:
            # Horizontal flip for mirror image matching
            image = cv2.flip(image, 1)
        
        return image
    
    def _detect_blue_boundary(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect blue dotted boundary in floor plan
        Returns (x, y, w, h) bounding box of the content area inside blue boundary
        """
        try:
            # Convert to HSV for better blue detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Blue color range (adjust for dotted blue lines)
            # Blue in HSV: H=100-130, S=50-255, V=50-255
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([140, 255, 255])
            
            # Create mask for blue color
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Find contours
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest blue contour (should be the boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add small margin inside the boundary
            margin = 10
            x += margin
            y += margin
            w -= 2 * margin
            h -= 2 * margin
            
            # Validate bounding box
            if w > 100 and h > 100 and w < image.shape[1] * 0.95 and h < image.shape[0] * 0.95:
                return (x, y, w, h)
            
            return None
            
        except Exception as e:
            logger.warning(f"Blue boundary detection failed: {e}")
            return None
    
    def _crop_to_boundary(self, image: np.ndarray) -> np.ndarray:
        """Crop image to blue boundary if detected, otherwise return original"""
        boundary = self._detect_blue_boundary(image)
        
        if boundary:
            x, y, w, h = boundary
            cropped = image[y:y+h, x:x+w]
            logger.debug(f"Cropped to blue boundary: {w}x{h} from {image.shape[1]}x{image.shape[0]}")
            return cropped
        else:
            logger.debug("No blue boundary detected, using full image")
            return image
    
    def _extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ORB features from image using GPU if available"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if self.cuda_available:
            # Upload to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(gray)
            
            # Detect and compute on GPU
            gpu_keypoints, gpu_descriptors = self.orb.detectAndComputeAsync(gpu_image, None)
            
            # Download results
            keypoints = self.orb.convert(gpu_keypoints)
            descriptors = gpu_descriptors.download() if gpu_descriptors is not None else None
        else:
            # CPU fallback
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def _compute_phash(self, img: np.ndarray, hash_size=16) -> imagehash.ImageHash:
        """Compute perceptual hash from numpy array"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(img_rgb)
        return imagehash.phash(pil_img, hash_size=16)
    
    def _hash_similarity(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """Compute similarity from hash distance"""
        distance = hash1 - hash2
        max_distance = len(hash1.hash) ** 2
        similarity = 1.0 - (distance / max_distance)
        return similarity
    
    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Match features using BFMatcher
        Returns similarity score 0-1
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        try:
            if self.cuda_available:
                # GPU matching
                gpu_desc1 = cv2.cuda_GpuMat()
                gpu_desc2 = cv2.cuda_GpuMat()
                gpu_desc1.upload(desc1)
                gpu_desc2.upload(desc2)
                
                matches = self.matcher.knnMatch(gpu_desc1, gpu_desc2, k=2)
            else:
                # CPU matching
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Calculate similarity score
            if len(desc1) > 0:
                similarity = len(good_matches) / min(len(desc1), len(desc2))
                return min(similarity, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Feature matching failed: {e}")
            return 0.0
    
    def _load_image_unicode_safe(self, img_path: Path) -> Optional[np.ndarray]:
        """Load image with Unicode support"""
        try:
            # Use imdecode for Unicode path support
            with open(str(img_path), 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            return None
    
    def _load_reference_features(self) -> Dict:
        """
        Load and extract features for all reference images
        Pre-generates augmented versions (mirror, slight rotations) for better matching
        """
        reference_data = {}
        
        for unit_type_dir in sorted(self.reference_dir.iterdir()):
            if not unit_type_dir.is_dir():
                continue
            
            unit_type = unit_type_dir.name
            reference_data[unit_type] = {
                'facade': {'modern': [], 'traditional': []},
                'floor_plan': []
            }
            
            # Load facade images (no augmentation for facades - they're fixed orientation)
            facade_dir = unit_type_dir / "Facade"
            if facade_dir.exists():
                for style in ['modern', 'traditional']:
                    style_dir = facade_dir / style
                    if style_dir.exists():
                        for img_path in style_dir.glob("*.jpg"):
                            img = self._load_image_unicode_safe(img_path)
                            if img is not None:
                                keypoints, descriptors = self._extract_features(img)
                                phash = self._compute_phash(img)
                                reference_data[unit_type]['facade'][style].append({
                                    'path': str(img_path),
                                    'name': img_path.name,
                                    'keypoints': keypoints,
                                    'descriptors': descriptors,
                                    'hash': phash,
                                    'augmented': False
                                })
            
            # Load floor plan images WITH AUGMENTATION
            floor_dir = unit_type_dir / "Floor Plans"
            if floor_dir.exists():
                for img_path in floor_dir.glob("*.jpg"):
                    img = self._load_image_unicode_safe(img_path)
                    if img is not None:
                        # Original image
                        keypoints, descriptors = self._extract_features(img)
                        phash = self._compute_phash(img)
                        reference_data[unit_type]['floor_plan'].append({
                            'path': str(img_path),
                            'name': img_path.name,
                            'keypoints': keypoints,
                            'descriptors': descriptors,
                            'hash': phash,
                            'augmented': False
                        })
                        
                        # Mirror image augmentation (for floor plans that can be flipped)
                        mirrored_img = cv2.flip(img, 1)
                        keypoints_m, descriptors_m = self._extract_features(mirrored_img)
                        phash_m = self._compute_phash(mirrored_img)
                        reference_data[unit_type]['floor_plan'].append({
                            'path': str(img_path),
                            'name': f"{img_path.stem}_mirror{img_path.suffix}",
                            'keypoints': keypoints_m,
                            'descriptors': descriptors_m,
                            'hash': phash_m,
                            'augmented': True,
                            'augmentation_type': 'mirror'
                        })
        
        return reference_data
    
    def _compare_with_augmentation(self, test_desc, test_hash, ref_data, is_floor_plan=True):
        """
        Compare test image with reference using both normal and mirror images
        Returns best similarity score
        
        Args:
            test_desc: Test image descriptors
            test_hash: Test image pHash
            ref_data: Reference data dict
            is_floor_plan: True for floor plans, False for facades
        
        Returns:
            Best similarity score and whether it was mirrored
        """
        # Normal comparison
        feature_sim = self._match_features(test_desc, ref_data['descriptors'])
        hash_sim = self._hash_similarity(test_hash, ref_data['hash'])
        
        if is_floor_plan:
            # Floor plans: 50% feature + 50% hash with boost
            similarity = 0.5 * feature_sim + 0.5 * hash_sim
            if feature_sim > 0.4 and hash_sim > 0.4:
                similarity = min(1.0, similarity * 1.1)  # 10% boost
        else:
            # Facades: 30% feature + 70% hash
            similarity = 0.3 * feature_sim + 0.7 * hash_sim
        
        return similarity, False  # Normal image (not mirrored)
    
    def verify_floor_plan(self, floor_plan_image: np.ndarray, declared_unit_type: str) -> Dict:
        """
        Verify floor plan using GPU-accelerated feature matching + pHash
        Crops to blue boundary first for better matching (zoom in effect)
        Compares against pre-augmented reference database (includes mirror images)
        """
        # ZOOM IN: Crop to blue boundary to focus on actual floor plan content
        cropped_image = self._crop_to_boundary(floor_plan_image)
        
        # Extract features from test image
        test_kp, test_desc = self._extract_features(cropped_image)
        test_hash = self._compute_phash(cropped_image)
        
        best_match = {'unit_type': None, 'similarity': 0, 'ref_name': None, 'is_augmented': False}
        all_scores = {}
        
        # Compare against ALL references (including augmented versions)
        for unit_type, data in self.reference_data.items():
            max_similarity = 0
            best_ref_name = None
            is_augmented = False
            
            for ref_data in data['floor_plan']:
                # Hybrid scoring: 50% feature + 50% hash
                feature_sim = self._match_features(test_desc, ref_data['descriptors'])
                hash_sim = self._hash_similarity(test_hash, ref_data['hash'])
                
                similarity = 0.5 * feature_sim + 0.5 * hash_sim
                
                # Boost if both metrics agree
                if feature_sim > 0.4 and hash_sim > 0.4:
                    similarity = min(1.0, similarity * 1.1)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_ref_name = ref_data['name']
                    is_augmented = ref_data.get('augmented', False)
                
                if similarity > best_match['similarity']:
                    best_match = {
                        'unit_type': unit_type,
                        'similarity': similarity,
                        'ref_name': ref_data['name'],
                        'is_augmented': ref_data.get('augmented', False),
                        'augmentation_type': ref_data.get('augmentation_type', 'none')
                    }
            
            all_scores[unit_type] = max_similarity
        
        # Determine if it's a mirror match
        is_mirrored = best_match.get('augmentation_type') == 'mirror'
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'reference': best_match.get('ref_name'),
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'all_scores': all_scores,
            'mirrored': is_mirrored,
            'threshold': self.SIMILARITY_THRESHOLD
        }
    
    def verify_facade(self, facade_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify facade using GPU-accelerated matching"""
        # Extract features
        test_kp, test_desc = self._extract_features(facade_image)
        test_hash = self._compute_phash(facade_image)
        
        best_match = {'unit_type': None, 'similarity': 0, 'style': None, 'ref_name': None}
        all_scores = {}
        
        for unit_type, data in self.reference_data.items():
            for style in ['modern', 'traditional']:
                max_similarity = 0
                
                for ref_data in data['facade'][style]:
                    # Hybrid scoring: 30% feature + 70% pHash
                    # Facades are more consistent, pHash works better
                    feature_sim = self._match_features(test_desc, ref_data['descriptors'])
                    hash_sim = self._hash_similarity(test_hash, ref_data['hash'])
                    similarity = 0.3 * feature_sim + 0.7 * hash_sim
                    
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
            'reference': best_match.get('ref_name'),
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'all_scores': all_scores
        }
    
    def verify_floor_plans_multi(self, floor_images: Dict[str, np.ndarray], declared_unit_type: str) -> Dict:
        """Verify multiple floor plan pages"""
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
        
        # Vote for most common detected unit type
        detected_types = [r['detected'] for r in results.values()]
        most_common = max(set(detected_types), key=detected_types.count) if detected_types else None
        avg_similarity = total_similarity / count
        
        return {
            'detected': most_common,
            'similarity': avg_similarity,
            'is_correct': most_common == declared_unit_type,
            'details': results,
            'vote_count': detected_types.count(most_common) if most_common else 0,
            'total_pages': count
        }
    
    def verify_facades_multi(self, facade_images: Dict[str, np.ndarray], declared_unit_type: str) -> Dict:
        """Verify multiple facade pages"""
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
        
        # Vote for most common detected unit type
        detected_types = [r['detected'] for r in results.values()]
        most_common = max(set(detected_types), key=detected_types.count) if detected_types else None
        
        # Get most common style
        styles = [r['style'] for r in results.values()]
        most_common_style = max(set(styles), key=styles.count) if styles else None
        
        avg_similarity = total_similarity / count
        
        return {
            'detected': most_common,
            'similarity': avg_similarity,
            'style': most_common_style,
            'is_correct': most_common == declared_unit_type,
            'details': results,
            'vote_count': detected_types.count(most_common) if most_common else 0,
            'total_pages': count
        }
    
    def verify_unit_type_multi(self, facade_images: Dict[str, np.ndarray], 
                               floor_images: Dict[str, np.ndarray],
                               declared_unit_type: str) -> Dict:
        """Complete GPU-accelerated verification using multiple pages"""
        facade_result = self.verify_facades_multi(facade_images, declared_unit_type)
        floor_result = self.verify_floor_plans_multi(floor_images, declared_unit_type)
        
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
