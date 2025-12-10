"""
Enhanced Unit Type Verification System v2
------------------------------------------
Improvements:
1. Remove "snap attached" line from floor plans
2. Data augmentation for robust matching
3. Multiple model support (DINOv2, CLIP, original DINO)
4. Better preprocessing pipeline
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image

class UnitTypeVerifierV2:
    """Enhanced verifier with preprocessing and multiple model support"""
    
    def __init__(self, reference_dir: str = "reference_images/unit types", model_type: str = "dinov2"):
        self.reference_dir = Path(reference_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        self._load_model()
        self.reference_embeddings = self._load_reference_embeddings()
    
    def _load_model(self):
        """Load selected model with GPU support"""
        print(f"Loading {self.model_type.upper()} model on {self.device}...")
        
        if self.model_type == "dinov2":
            # DINOv2 - More recent, better performance
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = 384
                print(f"[OK] DINOv2 ViT-S/14 loaded (384-dim embeddings)")
            except Exception as e:
                print(f"[!] DINOv2 failed: {e}, falling back to DINOv1...")
                self.model_type = "dinov1"
                self._load_model()
                return
        
        elif self.model_type == "dinov1":
            # Original DINO v1
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = 384
            print(f"[OK] DINOv1 ViT-S/16 loaded (384-dim embeddings)")
        
        elif self.model_type == "clip":
            # CLIP (if available)
            try:
                import clip
                self.model, self.preprocess_clip = clip.load("ViT-B/32", device=self.device)
                self.embedding_dim = 512
                print(f"[OK] CLIP ViT-B/32 loaded (512-dim embeddings)")
                return  # CLIP has its own preprocessing
            except ImportError:
                print("[!] CLIP not installed, falling back to DINOv1...")
                self.model_type = "dinov1"
                self._load_model()
                return
        
        # Standard preprocessing for DINO models
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _remove_snap_line(self, img: np.ndarray, is_floor_plan: bool = False) -> np.ndarray:
        """
        Remove 'snap attached' line from floor plan images
        This line appears in ground truth but not in test PDFs, causing mismatches
        """
        if not is_floor_plan:
            return img
        
        # The "snap attached" line typically appears at the bottom
        # Strategy: Detect horizontal text lines and remove bottom-most one
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find bottom-most horizontal line
            bottom_y = 0
            bottom_contour = None
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if y > bottom_y and w > img.shape[1] * 0.3:  # At least 30% width
                    bottom_y = y
                    bottom_contour = cnt
            
            if bottom_contour is not None:
                x, y, w, h = cv2.boundingRect(bottom_contour)
                # Crop out bottom section (add some margin)
                crop_y = max(0, y - 20)
                img = img[:crop_y, :]
        
        return img
    
    def _preprocess_image(self, img: np.ndarray, is_floor_plan: bool = False, augment: bool = False) -> List[torch.Tensor]:
        """
        Preprocess image with optional augmentation
        Returns list of tensors (original + augmented versions)
        """
        # Remove snap line if floor plan
        img = self._remove_snap_line(img, is_floor_plan)
        
        # Convert to PIL
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(img)
        
        tensors = []
        
        # Original image
        if self.model_type == "clip":
            tensors.append(self.preprocess_clip(img_pil).unsqueeze(0))
        else:
            tensors.append(self.preprocess(img_pil).unsqueeze(0))
        
        # Data augmentation if requested
        if augment:
            from torchvision import transforms
            augmentations = [
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomRotation(5),  # Slight rotation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            ]
            
            for aug in augmentations:
                if self.model_type != "clip":  # Skip augmentation for CLIP
                    tensors.append(aug(img_pil).unsqueeze(0))
        
        return tensors
    
    def _extract_embedding(self, img: np.ndarray, is_floor_plan: bool = False, use_augmentation: bool = True) -> np.ndarray:
        """
        Extract embedding with optional augmentation (average multiple views)
        """
        tensors = self._preprocess_image(img, is_floor_plan, augment=use_augmentation)
        
        embeddings = []
        with torch.no_grad():
            for tensor in tensors:
                tensor = tensor.to(self.device)
                
                if self.model_type == "clip":
                    embedding = self.model.encode_image(tensor)
                else:
                    embedding = self.model(tensor)
                
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().numpy().flatten())
        
        # Average all embeddings (original + augmented)
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Re-normalize
        
        return avg_embedding
    
    def _load_image_unicode_safe(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Load image with Unicode path support (Windows compatibility)
        cv2.imread() fails with Unicode paths on Windows
        """
        try:
            # Method 1: Use PIL then convert to numpy
            from PIL import Image
            pil_img = Image.open(str(img_path))
            img_array = np.array(pil_img)
            # Convert RGB to BGR for OpenCV compatibility
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            return img_bgr
        except Exception as e:
            print(f"[!] Failed to load {img_path.name}: {e}")
            return None
    
    def _load_reference_embeddings(self) -> Dict:
        """Load and compute embeddings for all reference images"""
        print("Loading reference images...")
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
            
            # Load facade images (modern/traditional)
            facade_dir = unit_type_dir / "Facade"
            if facade_dir.exists():
                for style in ['modern', 'traditional']:
                    style_dir = facade_dir / style
                    if style_dir.exists():
                        for img_path in style_dir.glob("*.jpg"):
                            img = self._load_image_unicode_safe(img_path)
                            if img is not None:
                                embedding = self._extract_embedding(img, is_floor_plan=False, use_augmentation=False)
                                reference_data[unit_type]['facade'][style].append({
                                    'path': str(img_path),
                                    'embedding': embedding
                                })
            
            # Load floor plan images
            floor_dir = unit_type_dir / "Floor Plans"
            if floor_dir.exists():
                for img_path in floor_dir.glob("*.jpg"):
                    img = self._load_image_unicode_safe(img_path)
                    if img is not None:
                        # IMPORTANT: Process floor plans with snap line removal
                        embedding = self._extract_embedding(img, is_floor_plan=True, use_augmentation=False)
                        reference_data[unit_type]['floor_plan'].append({
                            'path': str(img_path),
                            'embedding': embedding
                        })
        
        print(f"[OK] Loaded reference embeddings for {len(reference_data)} unit types")
        return reference_data
    
    def verify_facade(self, facade_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify facade against all unit types"""
        test_embedding = self._extract_embedding(facade_image, is_floor_plan=False, use_augmentation=True)
        
        best_match = {'unit_type': None, 'similarity': 0, 'style': None}
        all_scores = {}
        
        for unit_type, data in self.reference_embeddings.items():
            for style in ['modern', 'traditional']:
                style_embeddings = data['facade'][style]
                if style_embeddings:
                    similarities = [
                        np.dot(test_embedding, ref['embedding'])
                        for ref in style_embeddings
                    ]
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    
                    key = f"{unit_type}_{style}"
                    all_scores[key] = {'avg': avg_similarity, 'max': max_similarity}
                    
                    if max_similarity > best_match['similarity']:
                        best_match = {
                            'unit_type': unit_type,
                            'similarity': max_similarity,
                            'style': style,
                            'avg_similarity': avg_similarity
                        }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'avg_similarity': best_match.get('avg_similarity', best_match['similarity']),
            'style': best_match['style'],
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'all_scores': all_scores
        }
    
    def verify_floor_plan(self, floor_plan_image: np.ndarray, declared_unit_type: str) -> Dict:
        """Verify floor plan against all unit types"""
        # Use augmentation for test image, snap line removal is automatic
        test_embedding = self._extract_embedding(floor_plan_image, is_floor_plan=True, use_augmentation=True)
        
        best_match = {'unit_type': None, 'similarity': 0}
        all_scores = {}
        
        for unit_type, data in self.reference_embeddings.items():
            floor_embeddings = data['floor_plan']
            if floor_embeddings:
                similarities = [
                    np.dot(test_embedding, ref['embedding'])
                    for ref in floor_embeddings
                ]
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                all_scores[unit_type] = {'avg': avg_similarity, 'max': max_similarity}
                
                if max_similarity > best_match['similarity']:
                    best_match = {
                        'unit_type': unit_type,
                        'similarity': max_similarity,
                        'avg_similarity': avg_similarity
                    }
        
        return {
            'detected': best_match['unit_type'],
            'similarity': best_match['similarity'],
            'avg_similarity': best_match.get('avg_similarity', best_match['similarity']),
            'is_correct': best_match['unit_type'] == declared_unit_type,
            'all_scores': all_scores
        }
    
    def verify_unit_type(self, facade_image: np.ndarray, floor_plan_image: np.ndarray, 
                        declared_unit_type: str) -> Dict:
        """Complete verification: facade + floor plan"""
        facade_result = self.verify_facade(facade_image, declared_unit_type)
        floor_result = self.verify_floor_plan(floor_plan_image, declared_unit_type)
        
        # Combined decision
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


# Test function
def test_enhanced_verifier():
    """Test the enhanced verifier"""
    import glob
    
    print("\n" + "="*70)
    print("TESTING ENHANCED UNIT TYPE VERIFIER V2")
    print("="*70)
    
    # Try DINOv2 first, fall back to DINOv1
    try:
        verifier = UnitTypeVerifierV2(model_type="dinov2")
    except:
        print("[!] DINOv2 not available, using DINOv1...")
        verifier = UnitTypeVerifierV2(model_type="dinov1")
    
    # Test on a few PDFs
    test_pdfs = glob.glob("input_data/*.pdf")[:3]
    
    for pdf_path in test_pdfs:
        print(f"\n{'='*70}")
        print(f"Testing: {os.path.basename(pdf_path)}")
        print(f"{'='*70}")
        
        # Extract images (simplified - assume pages already extracted)
        # In real usage, this comes from pdf_extractor
        from pdf2image import convert_from_path
        
        poppler_path = r"C:\MEHBOOB HD\Roshn Backup\Demarcation_Verifier\version5\poppler\Library\bin"
        images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
        
        if len(images) >= 3:
            facade_img = cv2.cvtColor(np.array(images[1]), cv2.COLOR_RGB2BGR)  # Page 2
            floor_img = cv2.cvtColor(np.array(images[2]), cv2.COLOR_RGB2BGR)  # Page 3
            
            # Assume unit type from filename (simplified)
            # In real usage, this comes from ground truth
            declared_type = "V2C"  # Placeholder
            
            result = verifier.verify_unit_type(facade_img, floor_img, declared_type)
            
            print(f"\nDeclared: {declared_type}")
            print(f"Facade detected: {result['facade_verification']['detected']} "
                  f"({result['facade_verification']['similarity']:.1%}, "
                  f"{result['facade_verification']['style']})")
            print(f"Floor detected: {result['floor_plan_verification']['detected']} "
                  f"({result['floor_plan_verification']['similarity']:.1%})")
            print(f"Verdict: {result['verdict']} (avg similarity: {result['average_similarity']:.1%})")


if __name__ == "__main__":
    test_enhanced_verifier()
