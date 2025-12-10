"""
Unit Type Verification System
------------------------------
Verifies if extracted facade and floor plans match the declared unit type
using DINO embeddings and cosine similarity.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Use DINO directly (already working in facade detection)
try:
    from transformers import ViTImageProcessor, ViTModel
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    print("⚠ transformers not available, trying torch hub...")
    try:
        import torch
        DINO_AVAILABLE = True
    except ImportError:
        DINO_AVAILABLE = False

class UnitTypeVerifier:
    """Verifies unit types using ground truth facade and floor plan images"""
    
    def __init__(self, reference_dir: str = "reference_images/unit types"):
        self.reference_dir = Path(reference_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load DINO v1 model (same as facade detection - works with GPU)
        print("Loading DINO ViT-S/16 model for unit type verification...")
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.model.to(self.device)
        self.model.eval()
        
        # DINO preprocessing (standard ImageNet normalization)
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"[OK] DINO model loaded on {self.device}")
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings()
        print(f"[OK] Loaded reference embeddings for {len(self.reference_embeddings)} unit types")
    
    def _load_reference_embeddings(self) -> Dict[str, Dict]:
        """Load and cache reference image embeddings"""
        embeddings = {}
        
        if not self.reference_dir.exists():
            print(f"❌ Reference directory not found: {self.reference_dir}")
            return embeddings
        
        # Scan all unit type folders
        for unit_folder in self.reference_dir.iterdir():
            if not unit_folder.is_dir():
                continue
            
            unit_type = unit_folder.name
            print(f"  Loading {unit_type}...")
            
            embeddings[unit_type] = {
                'facade_traditional': [],
                'facade_modern': [],
                'floor_plans': []
            }
            
            # Load facade images
            facade_dir = unit_folder / "Facade"
            if facade_dir.exists():
                # Traditional facades
                trad_dir = facade_dir / "traditional"
                if trad_dir.exists():
                    for img_path in trad_dir.glob("*.jpg"):
                        emb = self._extract_embedding(str(img_path))
                        if emb is not None:
                            embeddings[unit_type]['facade_traditional'].append({
                                'embedding': emb,
                                'path': str(img_path)
                            })
                
                # Modern facades
                modern_dir = facade_dir / "modern"
                if modern_dir.exists():
                    for img_path in modern_dir.glob("*.jpg"):
                        emb = self._extract_embedding(str(img_path))
                        if emb is not None:
                            embeddings[unit_type]['facade_modern'].append({
                                'embedding': emb,
                                'path': str(img_path)
                            })
            
            # Load floor plans
            floor_dir = unit_folder / "Floor Plans"
            if floor_dir.exists():
                for img_path in floor_dir.glob("*.jpg"):
                    emb = self._extract_embedding(str(img_path))
                    if emb is not None:
                        embeddings[unit_type]['floor_plans'].append({
                            'embedding': emb,
                            'path': str(img_path)
                        })
        
        return embeddings
    
    def _extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract DINO embedding from image"""
        try:
            from PIL import Image
            
            img = Image.open(image_path).convert('RGB')
            
            # Apply DINO preprocessing
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
        except Exception as e:
            print(f"    Error extracting embedding from {image_path}: {e}")
            return None
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def verify_facade(self, facade_image: np.ndarray, declared_unit_type: str, 
                     facade_style: str = None) -> Dict:
        """
        Verify if facade matches declared unit type
        
        Args:
            facade_image: Extracted facade image (numpy array)
            declared_unit_type: Unit type from Excel (e.g., "C20", "V2C")
            facade_style: "traditional" or "modern" (optional, will try both if not specified)
        
        Returns:
            Dict with verification results
        """
        from PIL import Image
        
        # Convert numpy array to PIL Image
        if len(facade_image.shape) == 3 and facade_image.shape[2] == 3:
            facade_rgb = cv2.cvtColor(facade_image, cv2.COLOR_BGR2RGB)
        else:
            facade_rgb = facade_image
        
        pil_image = Image.fromarray(facade_rgb.astype('uint8'), 'RGB')
        
        # Apply DINO preprocessing
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            test_embedding = self.model(img_tensor).cpu().numpy().flatten()
        
        # Find best match across all unit types
        best_match = None
        best_similarity = -1
        best_unit_type = None
        best_style = None
        
        for unit_type, refs in self.reference_embeddings.items():
            # Check traditional facades
            if facade_style is None or facade_style == "traditional":
                for ref in refs['facade_traditional']:
                    sim = self._cosine_similarity(test_embedding, ref['embedding'])
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = ref
                        best_unit_type = unit_type
                        best_style = "traditional"
            
            # Check modern facades
            if facade_style is None or facade_style == "modern":
                for ref in refs['facade_modern']:
                    sim = self._cosine_similarity(test_embedding, ref['embedding'])
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = ref
                        best_unit_type = unit_type
                        best_style = "modern"
        
        # Determine if match is correct
        is_correct = (best_unit_type == declared_unit_type)
        
        return {
            'declared': declared_unit_type,
            'detected': best_unit_type,
            'similarity': float(best_similarity),
            'style': best_style,
            'is_correct': is_correct,
            'match_path': best_match['path'] if best_match else None,
            'confidence': 'high' if best_similarity > 0.85 else 'medium' if best_similarity > 0.70 else 'low'
        }
    
    def verify_floor_plan(self, floor_plan_image: np.ndarray, declared_unit_type: str) -> Dict:
        """
        Verify if floor plan matches declared unit type
        
        Args:
            floor_plan_image: Extracted floor plan image (numpy array)
            declared_unit_type: Unit type from Excel
        
        Returns:
            Dict with verification results
        """
        from PIL import Image
        
        # Convert numpy array to PIL Image
        if len(floor_plan_image.shape) == 3 and floor_plan_image.shape[2] == 3:
            floor_rgb = cv2.cvtColor(floor_plan_image, cv2.COLOR_BGR2RGB)
        else:
            floor_rgb = floor_plan_image
        
        pil_image = Image.fromarray(floor_rgb.astype('uint8'), 'RGB')
        
        # Apply DINO preprocessing
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            test_embedding = self.model(img_tensor).cpu().numpy().flatten()
        
        # Find best match across all unit types
        best_match = None
        best_similarity = -1
        best_unit_type = None
        
        for unit_type, refs in self.reference_embeddings.items():
            for ref in refs['floor_plans']:
                sim = self._cosine_similarity(test_embedding, ref['embedding'])
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = ref
                    best_unit_type = unit_type
        
        # Determine if match is correct
        is_correct = (best_unit_type == declared_unit_type)
        
        return {
            'declared': declared_unit_type,
            'detected': best_unit_type,
            'similarity': float(best_similarity),
            'is_correct': is_correct,
            'match_path': best_match['path'] if best_match else None,
            'confidence': 'high' if best_similarity > 0.85 else 'medium' if best_similarity > 0.70 else 'low'
        }
    
    def verify_unit_type(self, facade_image: np.ndarray, floor_plan_image: np.ndarray,
                        declared_unit_type: str, facade_style: str = None) -> Dict:
        """
        Complete unit type verification (facade + floor plan)
        
        Returns:
            Dict with combined verification results
        """
        facade_result = self.verify_facade(facade_image, declared_unit_type, facade_style)
        floor_result = self.verify_floor_plan(floor_plan_image, declared_unit_type)
        
        # Overall verdict
        both_correct = facade_result['is_correct'] and floor_result['is_correct']
        avg_similarity = (facade_result['similarity'] + floor_result['similarity']) / 2
        
        return {
            'declared_unit_type': declared_unit_type,
            'facade_verification': facade_result,
            'floor_plan_verification': floor_result,
            'overall_correct': both_correct,
            'average_similarity': float(avg_similarity),
            'verdict': 'PASS' if both_correct else 'FAIL',
            'details': self._format_verdict_details(facade_result, floor_result)
        }
    
    def _format_verdict_details(self, facade_result: Dict, floor_result: Dict) -> str:
        """Format human-readable verdict details"""
        details = []
        
        # Facade
        facade_status = "[OK]" if facade_result['is_correct'] else "[X]"
        details.append(f"{facade_status} Facade: {facade_result['declared']} → {facade_result['detected']} "
                      f"({facade_result['similarity']:.2%}, {facade_result['style']})")
        
        # Floor plan
        floor_status = "[OK]" if floor_result['is_correct'] else "[X]"
        details.append(f"{floor_status} Floor Plan: {floor_result['declared']} → {floor_result['detected']} "
                      f"({floor_result['similarity']:.2%})")
        
        return " | ".join(details)


def test_unit_verifier():
    """Test the unit type verifier on sample images"""
    print("=" * 80)
    print("UNIT TYPE VERIFICATION TEST")
    print("=" * 80)
    
    verifier = UnitTypeVerifier()
    
    # Test on input_data PDFs
    import fitz
    from pathlib import Path
    
    input_dir = Path("input_data")
    test_pdfs = list(input_dir.glob("*.pdf"))[:2]  # Test first 2 PDFs
    
    for pdf_path in test_pdfs:
        print(f"\n{'='*80}")
        print(f"Testing: {pdf_path.name}")
        print(f"{'='*80}")
        
        try:
            doc = fitz.open(str(pdf_path))
            
            # Extract facade (page 2) and floor plan (page 3)
            if len(doc) < 3:
                print(f"  ⚠ PDF has only {len(doc)} pages, skipping...")
                continue
            
            facade_page = doc[1]  # Page 2
            floor_page = doc[2]   # Page 3
            
            # Convert to images
            mat = fitz.Matrix(2, 2)  # 2x zoom
            facade_pix = facade_page.get_pixmap(matrix=mat)
            floor_pix = floor_page.get_pixmap(matrix=mat)
            
            # Convert to numpy arrays (RGB)
            facade_img = np.frombuffer(facade_pix.samples, dtype=np.uint8).reshape(
                facade_pix.height, facade_pix.width, facade_pix.n)
            floor_img = np.frombuffer(floor_pix.samples, dtype=np.uint8).reshape(
                floor_pix.height, floor_pix.width, floor_pix.n)
            
            # Convert RGBA to RGB if needed
            if facade_pix.n == 4:
                facade_img = cv2.cvtColor(facade_img, cv2.COLOR_RGBA2RGB)
            if floor_pix.n == 4:
                floor_img = cv2.cvtColor(floor_img, cv2.COLOR_RGBA2RGB)
            
            # Test against all unit types and find best match
            best_result = None
            best_avg = -1
            
            for unit_type in verifier.reference_embeddings.keys():
                result = verifier.verify_unit_type(facade_img, floor_img, unit_type)
                if result['average_similarity'] > best_avg:
                    best_avg = result['average_similarity']
                    best_result = result
            
            if best_result:
                print(f"\n  Best Match:")
                print(f"    Unit Type: {best_result['declared_unit_type']}")
                print(f"    Verdict: {best_result['verdict']}")
                print(f"    Details: {best_result['details']}")
                print(f"    Average Similarity: {best_result['average_similarity']:.2%}")
            
            doc.close()
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_unit_verifier()
