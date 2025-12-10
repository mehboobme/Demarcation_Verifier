"""Debug what's happening with page 8 facade detection"""
import sys
sys.path.insert(0, '.')

from pdf_extractor import PDFExtractor
from unit_type_verifier_phash import HybridUnitTypeVerifier
import cv2
import os

# Load verifier
print("Loading verifier...")
verifier = HybridUnitTypeVerifier()

# Extract PDF
pdf_path = 'input_data/DM1-D02-2A-29-526-01.pdf'
print(f"\nProcessing: {pdf_path}")
print("="*70)

extractor = PDFExtractor(pdf_path, vision_model=None)
data = extractor.extract_all()

# Get page 8 (front facade)
page8_image = data.facade_front_image

if page8_image is None:
    print("ERROR: Page 8 image not found!")
    sys.exit(1)

print(f"\nPage 8 image shape: {page8_image.shape}")

# Save the page 8 image for inspection
debug_path = "debug_page8_from_pdf.jpg"
cv2.imwrite(debug_path, page8_image)
print(f"Saved page 8 to: {debug_path}")

# Test against ALL unit types to see similarities
print("\n" + "="*70)
print("TESTING PAGE 8 AGAINST ALL UNIT TYPES")
print("="*70)

declared_type = 'V1A'
facade_style = extractor.data.facade_type  # T or M

print(f"\nDeclared type: {declared_type}")
print(f"Facade style detected: {facade_style} ({'traditional' if facade_style == 'T' else 'modern'})")

# Manually compute similarity against all reference images
facade_hash = verifier._compute_phash(page8_image)

print("\nSimilarities to ALL facade reference images:")
print("-" * 70)

all_results = []
for unit_type in verifier.reference_hashes.keys():
    if 'facade' not in verifier.reference_hashes[unit_type]:
        continue
    
    for style in ['modern', 'traditional']:
        if style not in verifier.reference_hashes[unit_type]['facade']:
            continue
        
        facade_hashes = verifier.reference_hashes[unit_type]['facade'][style]
        
        # facade_hashes is a list of dicts with 'hash' key
        for ref_data in facade_hashes:
            similarity = verifier._hash_similarity(facade_hash, ref_data['hash'])
            all_results.append({
                'unit_type': unit_type,
                'style': style,
                'ref_name': ref_data['name'],
                'similarity': similarity
            })

# Sort by similarity
all_results.sort(key=lambda x: x['similarity'], reverse=True)

# Show top 20 matches
print("\nTOP 20 MATCHES:")
output_lines = []
for i, result in enumerate(all_results[:20], 1):
    marker = " <-- DECLARED TYPE" if result['unit_type'] == declared_type else ""
    line = f"{i:2d}. {result['unit_type']:4s} {result['style']:11s} {result['similarity']:6.1%}{marker}"
    print(line)
    output_lines.append(line)

# Save to file for inspection
with open('debug_page8_results.txt', 'w', encoding='utf-8') as f:
    f.write("TOP 20 MATCHES:\n")
    for i, result in enumerate(all_results[:20], 1):
        marker = " <-- DECLARED TYPE" if result['unit_type'] == declared_type else ""
        f.write(f"{i:2d}. {result['unit_type']:4s} {result['style']:11s} {result['similarity']:6.1%} - {result['ref_name']}{marker}\n")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)

top_match = all_results[0]
print(f"\nTop match: {top_match['unit_type']} ({top_match['style']}) with {top_match['similarity']:.1%}")
print(f"Declared type (V1A) best match: ", end="")

# Find best V1A match
v1a_matches = [r for r in all_results if r['unit_type'] == 'V1A']
if v1a_matches:
    best_v1a = v1a_matches[0]
    print(f"{best_v1a['similarity']:.1%} (rank #{all_results.index(best_v1a) + 1})")
else:
    print("NOT FOUND in reference images!")

print("\n" + "="*70)
print("WHAT YOU SHOULD CHECK:")
print("="*70)
print("1. Look at the saved image: debug_page8_from_pdf.jpg")
print("2. What unit type did you replace page 8 with?")
print("3. Compare with reference images in:")
print("   reference_images/unit types/{unit_type}/Facade/{modern|traditional}/")
print(f"4. Top match is {top_match['unit_type']} - is this what you expected?")

if top_match['unit_type'] == 'V1A':
    print("\n[ISSUE] Page 8 still matches V1A best!")
    print("Possible reasons:")
    print("  - The replacement didn't save properly in the PDF")
    print("  - You replaced with a very similar facade")
    print("  - The PDF viewer/editor cached the old version")
else:
    print(f"\n[OK] Page 8 matches {top_match['unit_type']}, not V1A")
    print("But the voting system uses multi-page consensus.")
