"""
Explore better matching approaches for unit type verification
Test different algorithms to improve accuracy
"""
import sys
sys.path.insert(0, '.')

from pdf_extractor import PDFExtractor
from unit_type_verifier_phash import HybridUnitTypeVerifier
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

# Test on plot 526
pdf_path = 'input_data/DM1-D02-2A-29-526-01.pdf'
declared_type = 'V1A'

print("="*70)
print("EXPLORING BETTER MATCHING APPROACHES")
print("="*70)

# Extract PDF
extractor = PDFExtractor(pdf_path, vision_model=None)
data = extractor.extract_all()

# Get page 4 (ground floor)
page4 = data.ground_floor_image
if page4 is None:
    print("ERROR: Page 4 not found!")
    sys.exit(1)

print(f"\nTest image: Page 4 (Ground Floor)")
print(f"Declared type: {declared_type}")
print(f"Image shape: {page4.shape}")

# Load reference images
ref_dir = Path("reference_images/unit types")
reference_images = {}

print("\nLoading reference images...")
for unit_type_dir in ref_dir.iterdir():
    if not unit_type_dir.is_dir():
        continue
    
    unit_type = unit_type_dir.name
    floor_dir = unit_type_dir / "Floor Plans"
    
    if floor_dir.exists():
        reference_images[unit_type] = []
        for img_path in floor_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is not None:
                reference_images[unit_type].append({
                    'path': str(img_path),
                    'image': img,
                    'name': img_path.name
                })

print(f"Loaded {sum(len(v) for v in reference_images.values())} reference floor plans")

# Preprocessing function
def preprocess_image(img, target_size=(800, 600)):
    """Preprocess image for better matching"""
    # Resize
    resized = cv2.resize(img, target_size)
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Enhance edges
    edges = cv2.Canny(denoised, 50, 150)
    
    # Adaptive threshold for better structure extraction
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    return {
        'gray': gray,
        'denoised': denoised,
        'edges': edges,
        'thresh': thresh,
        'resized': resized
    }

# Preprocess test image
test_processed = preprocess_image(page4)

print("\n" + "="*70)
print("APPROACH 1: ENHANCED PHASH (current)")
print("="*70)

verifier = HybridUnitTypeVerifier()
test_hash = verifier._compute_phash(page4)

phash_results = []
for unit_type, refs in reference_images.items():
    for ref_data in refs:
        ref_pil = Image.fromarray(cv2.cvtColor(ref_data['image'], cv2.COLOR_BGR2RGB))
        ref_hash = imagehash.phash(ref_pil, hash_size=16)
        similarity = 1 - (test_hash - ref_hash) / 256.0
        phash_results.append({
            'unit_type': unit_type,
            'similarity': similarity,
            'name': ref_data['name']
        })

phash_results.sort(key=lambda x: x['similarity'], reverse=True)
print(f"\nTop 5 matches:")
for i, r in enumerate(phash_results[:5], 1):
    marker = " <-- DECLARED" if r['unit_type'] == declared_type else ""
    print(f"{i}. {r['unit_type']:4s} {r['similarity']:6.1%}{marker}")

print("\n" + "="*70)
print("APPROACH 2: SSIM (Structural Similarity) with preprocessing")
print("="*70)

ssim_results = []
for unit_type, refs in reference_images.items():
    for ref_data in refs:
        ref_processed = preprocess_image(ref_data['image'])
        
        # Compute SSIM on denoised grayscale
        similarity = ssim(test_processed['denoised'], ref_processed['denoised'])
        
        ssim_results.append({
            'unit_type': unit_type,
            'similarity': similarity,
            'name': ref_data['name']
        })

ssim_results.sort(key=lambda x: x['similarity'], reverse=True)
print(f"\nTop 5 matches:")
for i, r in enumerate(ssim_results[:5], 1):
    marker = " <-- DECLARED" if r['unit_type'] == declared_type else ""
    print(f"{i}. {r['unit_type']:4s} {r['similarity']:6.1%}{marker}")

print("\n" + "="*70)
print("APPROACH 3: ORB Feature Matching")
print("="*70)

orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Detect keypoints in test image
kp1, des1 = orb.detectAndCompute(test_processed['thresh'], None)

orb_results = []
for unit_type, refs in reference_images.items():
    for ref_data in refs:
        ref_processed = preprocess_image(ref_data['image'])
        kp2, des2 = orb.detectAndCompute(ref_processed['thresh'], None)
        
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Good matches (low distance)
            good_matches = [m for m in matches if m.distance < 50]
            similarity = len(good_matches) / max(len(kp1), len(kp2))
            
            orb_results.append({
                'unit_type': unit_type,
                'similarity': similarity,
                'good_matches': len(good_matches),
                'total_kp': len(kp1),
                'name': ref_data['name']
            })

orb_results.sort(key=lambda x: x['similarity'], reverse=True)
print(f"\nTop 5 matches:")
for i, r in enumerate(orb_results[:5], 1):
    marker = " <-- DECLARED" if r['unit_type'] == declared_type else ""
    print(f"{i}. {r['unit_type']:4s} {r['similarity']:6.1%} ({r['good_matches']} matches){marker}")

print("\n" + "="*70)
print("APPROACH 4: Edge-based Matching (Canny + Template)")
print("="*70)

edge_results = []
for unit_type, refs in reference_images.items():
    for ref_data in refs:
        ref_processed = preprocess_image(ref_data['image'])
        
        # Compare edge maps using template matching
        result = cv2.matchTemplate(test_processed['edges'], ref_processed['edges'], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        edge_results.append({
            'unit_type': unit_type,
            'similarity': max_val,
            'name': ref_data['name']
        })

edge_results.sort(key=lambda x: x['similarity'], reverse=True)
print(f"\nTop 5 matches:")
for i, r in enumerate(edge_results[:5], 1):
    marker = " <-- DECLARED" if r['unit_type'] == declared_type else ""
    print(f"{i}. {r['unit_type']:4s} {r['similarity']:6.1%}{marker}")

print("\n" + "="*70)
print("APPROACH 5: Hybrid (Weighted combination)")
print("="*70)

# Combine results with weights
hybrid_scores = {}
for unit_type in reference_images.keys():
    # Get average scores from each method
    phash_avg = np.mean([r['similarity'] for r in phash_results if r['unit_type'] == unit_type])
    ssim_avg = np.mean([r['similarity'] for r in ssim_results if r['unit_type'] == unit_type])
    orb_avg = np.mean([r['similarity'] for r in orb_results if r['unit_type'] == unit_type])
    edge_avg = np.mean([r['similarity'] for r in edge_results if r['unit_type'] == unit_type])
    
    # Weighted combination (adjust weights based on performance)
    weighted = (
        0.3 * phash_avg +  # pHash is fast and good for overall structure
        0.3 * ssim_avg +   # SSIM is good for structural similarity
        0.2 * orb_avg +    # ORB is good for feature matching
        0.2 * edge_avg     # Edge matching is good for layout
    )
    
    hybrid_scores[unit_type] = {
        'total': weighted,
        'phash': phash_avg,
        'ssim': ssim_avg,
        'orb': orb_avg,
        'edge': edge_avg
    }

hybrid_results = sorted(hybrid_scores.items(), key=lambda x: x[1]['total'], reverse=True)
print(f"\nTop 5 matches:")
for i, (unit_type, scores) in enumerate(hybrid_results[:5], 1):
    marker = " <-- DECLARED" if unit_type == declared_type else ""
    print(f"{i}. {unit_type:4s} {scores['total']:6.1%} (P:{scores['phash']:.1%} S:{scores['ssim']:.1%} O:{scores['orb']:.1%} E:{scores['edge']:.1%}){marker}")

print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

# Find which method got the declared type ranked highest
methods = {
    'pHash': next((i for i, r in enumerate(phash_results) if r['unit_type'] == declared_type), None),
    'SSIM': next((i for i, r in enumerate(ssim_results) if r['unit_type'] == declared_type), None),
    'ORB': next((i for i, r in enumerate(orb_results) if r['unit_type'] == declared_type), None),
    'Edge': next((i for i, r in enumerate(edge_results) if r['unit_type'] == declared_type), None),
    'Hybrid': next((i for i, (ut, _) in enumerate(hybrid_results) if ut == declared_type), None)
}

print(f"\nRank of declared type ({declared_type}) in each method:")
for method, rank in methods.items():
    if rank is not None:
        print(f"  {method:10s}: #{rank + 1}")
    else:
        print(f"  {method:10s}: Not found")

print(f"\nBest method for this case: ", end="")
best_method = min(methods.items(), key=lambda x: x[1] if x[1] is not None else 999)
print(f"{best_method[0]} (rank #{best_method[1] + 1})")
