"""
Unit Type Verification System V3
---------------------------------
Refined accuracy improvements:
1. Advanced preprocessing (signature removal, noise reduction)
2. Multiple similarity metrics (cosine + euclidean + SSIM hybrid)
3. Perceptual hashing for exact match detection
4. Region-based comparison (ignore margins/stamps)
5. Adaptive thresholding
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class UnitTypeVerifierV3:
    """Enhanced verifier with advanced preprocessing and multi-metric comparison"""
    
    def __init__(self, reference_dir: str = "reference_images/unit types", model_type: str = "dinov2"):
        self.reference_dir = Path(reference_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        self._load_model()
        self.reference_embeddings = self._load_reference_embeddings()
    
    def _load_model(self):
        """Load DINOv2 with fallback"""
        print(f"Loading {self.model_type.upper()} model on {self.device}...")
        
        if self.model_type == "dinov2":
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                self.model.to(self.device)
                self.model.eval()
                print(f"[OK] DINOv2 ViT-S/14 loaded")
            except Exception as e:
                print(f"[!] DINOv2 failed: {e}, using DINOv1...")
                self.model_type = "dinov1"
                self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
                self.model.to(self.device)
                self.model.eval()
                print(f"[OK] DINOv1 ViT-S/16 loaded")
        
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _advanced_preprocessing(self, img: np.ndarray, is_floor_plan: bool = False) -> np.ndarray:
        """
        Advanced preprocessing to handle:
        - Signature removal areas (white boxes on PDFs)
        - Scan artifacts (noise, compression)
        - Margin differences
        - Resolution variations
        """
        # Convert to grayscale for preprocessing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 1. Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Detect and fill white regions (likely signature removal boxes)
        _, thresh = cv2.threshold(denoised, 250, 255, cv2.THRESH_BINARY)
        white_regions = thresh == 255
        
        # Fill white regions with surrounding average (inpainting)
        mask = white_regions.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        inpainted = cv2.inpaint(denoised, mask_dilated, 3, cv2.INPAINT_TELEA)
        
        # 3. Adaptive histogram equalization (normalize contrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(inpainted)
        
        # 4. Edge enhancement (architectural drawings have strong lines)
        edges = cv2.Canny(equalized, 50, 150)
        edge_enhanced = cv2.addWeighted(equalized, 0.7, edges, 0.3, 0)
        
        # 5. Remove text at bottom (snap line, stamps, etc.)
        if is_floor_plan:
            h, w = edge_enhanced.shape
            crop_h = int(h * 0.92)  # Remove bottom 8%
            edge_enhanced = edge_enhanced[:crop_h, :]
        
        # Convert back to BGR for consistency
        if len(img.shape) == 3:
            processed = cv2.cvtColor(edge_enhanced, cv2.COLOR_GRAY2BGR)
        else:
            processed = edge_enhanced
        
        return processed
    
    def _crop_to_content(self, img: np.ndarray, margin_percent: float = 0.05) -> np.ndarray:
        """
        Crop image to actual content, removing empty margins
        Helps when reference and test images have different margins
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Threshold to find content
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find bounding box of content
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add small margin
            margin_x = int(w * margin_percent)
            margin_y = int(h * margin_percent)
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(img.shape[1] - x, w + 2 * margin_x)
            h = min(img.shape[0] - y, h + 2 * margin_y)
            
            return img[y:y+h, x:x+w]
        
        return img
    
    def _compute_perceptual_hash(self, img: np.ndarray, hash_size: int = 16) -> np.ndarray:
        """
        Compute perceptual hash for fast similarity check
        Good for detecting near-identical images
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Resize to hash size
        resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        
        # Compute DCT
        dct = cv2.dct(np.float32(resized))
        
        # Keep top-left 8x8 (low frequencies)
        dct_low = dct[:8, :8]
        
        # Compute median
        median = np.median(dct_low)
        
        # Binary hash
        hash_val = dct_low > median
        
        return hash_val.flatten()
    
    def _hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Compute normalized Hamming distance between two hashes"""
        return np.sum(hash1 != hash2) / len(hash1)
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index
        Better than pixel-wise comparison for structural content
        """
        # Resize to same dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_size = (min(w1, w2), min(h1, h2))
        
        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)
        
        # Convert to grayscale
        if len(img1_resized.shape) == 3:
            img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1_resized
        
        if len(img2_resized.shape) == 3:
            img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2_resized
        
        # Compute SSIM
        score, _ = ssim(img1_gray, img2_gray, full=True)
        return score
    
    def _load_image_unicode_safe(self, img_path: Path) -> Optional[np.ndarray]:
        """Load image with Unicode support"""
        try:
            pil_img = Image.open(str(img_path))
            img_array = np.array(pil_img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            return img_bgr
        except Exception as e:
            return None
    
    def _extract_embedding(self, img: np.ndarray, use_preprocessing: bool = True) -> np.ndarray:
        """Extract DINO embedding with optional advanced preprocessing"""
        if use_preprocessing:
            # Apply advanced preprocessing
            processed = self._advanced_preprocessing(img)
            processed = self._crop_to_content(processed)
        else:
            processed = img
        
        # Convert to PIL
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(processed)
        
        # Extract DINO embedding
        tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    
    def _load_reference_embeddings(self) -> Dict:
        """Load reference images and compute multi-metric features"""
        print("Loading reference images with advanced features...")
        reference_data = {}
        
        for unit_type_dir in sorted(self.reference_dir.iterdir()):
            if not unit_type_dir.is_dir():
                continue
            
            unit_type = unit_type_dir.name
            print(f"  Loading {unit_type}...")
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
                                # Compute multiple features
                                embedding = self._extract_embedding(img, use_preprocessing=True)
                                phash = self._compute_perceptual_hash(img)
                                preprocessed = self._advanced_preprocessing(img, is_floor_plan=False)
                                preprocessed = self._crop_to_content(preprocessed)
                                
                                reference_data[unit_type]['facade'][style].append({
                                    'path': str(img_path),
                                    'embedding': embedding,
                                    'phash': phash,
                                    'preprocessed_img': preprocessed
                                })
            
            # Load floor plan images
            floor_dir = unit_type_dir / "Floor Plans"
            if floor_dir.exists():
                for img_path in floor_dir.glob("*.jpg"):
                    img = self._load_image_unicode_safe(img_path)
                    if img is not None:
                        embedding = self._extract_embedding(img, use_preprocessing=True)
                        phash = self._compute_perceptual_hash(img)
                        preprocessed = self._advanced_preprocessing(img, is_floor_plan=True)
                        preprocessed = self._crop_to_content(preprocessed)
                        
                        reference_data[unit_type]['floor_plan'].append({
                            'path': str(img_path),
                            'embedding': embedding,
                            'phash': phash,
                            'preprocessed_img': preprocessed
                        })
        
        print(f"[OK] Loaded reference embeddings for {len(reference_data)} unit types")
        return reference_data
    
    def _multi_metric_similarity(self, test_img: np.ndarray, ref_data: dict, 
                                  is_floor_plan: bool = False) -> Dict:
        """
        Compute similarity using multiple metrics and combine
        Returns: combined score with individual metric breakdowns
        """
        # Preprocess test image
        test_preprocessed = self._advanced_preprocessing(test_img, is_floor_plan=is_floor_plan)
        test_preprocessed = self._crop_to_content(test_preprocessed)
        
        # Extract features
        test_embedding = self._extract_embedding(test_img, use_preprocessing=True)
        test_phash = self._compute_perceptual_hash(test_img)
        
        # Compute similarities
        cosine_sim = np.dot(test_embedding, ref_data['embedding'])
        
        # Hamming distance for phash (convert to similarity)
        hamming_dist = self._hamming_distance(test_phash, ref_data['phash'])
        phash_sim = 1.0 - hamming_dist
        
        # SSIM for structural similarity
        ssim_score = self._compute_ssim(test_preprocessed, ref_data['preprocessed_img'])
        
        # Combined score (weighted average)
        # DINO: 50%, SSIM: 30%, pHash: 20%
        combined = 0.5 * cosine_sim + 0.3 * ssim_score + 0.2 * phash_sim
        
        return {
            'combined': combined,
            'cosine': cosine_sim,
            'ssim': ssim_score,
            'phash': phash_sim
        }
    
    def verify_facade(self, facade_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify facade with multi-metric comparison"""
        best_match = {'unit_type': None, 'similarity': 0, 'style': None, 'metrics': {}}
        all_scores = {}
        
        for unit_type, data in self.reference_embeddings.items():
            for style in ['modern', 'traditional']:
                style_refs = data['facade'][style]
                if style_refs:
                    scores = []
                    for ref in style_refs:
                        metrics = self._multi_metric_similarity(facade_image, ref, is_floor_plan=False)
                        scores.append(metrics)
                    
                    # Take best score
                    best_score = max(scores, key=lambda x: x['combined'])
                    avg_combined = np.mean([s['combined'] for s in scores])
                    
                    key = f"{unit_type}_{style}"
                    all_scores[key] = {
                        'max': best_score['combined'],
                        'avg': avg_combined,
                        'metrics': best_score
                    }
                    
                    if best_score['combined'] > best_match['similarity']:
                        best_match = {
                            'unit_type': unit_type,
                            'similarity': best_score['combined'],
                            'style': style,
                            'metrics': best_score
                        }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'style': best_match['style'],
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'metrics': best_match['metrics'],
            'all_scores': all_scores
        }
    
    def verify_floor_plan(self, floor_plan_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify floor plan with multi-metric comparison"""
        best_match = {'unit_type': None, 'similarity': 0, 'metrics': {}}
        all_scores = {}
        
        for unit_type, data in self.reference_embeddings.items():
            floor_refs = data['floor_plan']
            if floor_refs:
                scores = []
                for ref in floor_refs:
                    metrics = self._multi_metric_similarity(floor_plan_image, ref, is_floor_plan=True)
                    scores.append(metrics)
                
                best_score = max(scores, key=lambda x: x['combined'])
                avg_combined = np.mean([s['combined'] for s in scores])
                
                all_scores[unit_type] = {
                    'max': best_score['combined'],
                    'avg': avg_combined,
                    'metrics': best_score
                }
                
                if best_score['combined'] > best_match['similarity']:
                    best_match = {
                        'unit_type': unit_type,
                        'similarity': best_score['combined'],
                        'metrics': best_score
                    }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'metrics': best_match['metrics'],
            'all_scores': all_scores
        }
    
    def verify_unit_type(self, facade_image: np.ndarray, floor_plan_image: np.ndarray,
                        declared_unit_type: str) -> Dict:
        """Complete verification with multi-metric approach"""
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


if __name__ == "__main__":
    print("Unit Type Verifier V3 - Multi-Metric Approach")
    print("Advanced preprocessing + DINO + SSIM + pHash")
