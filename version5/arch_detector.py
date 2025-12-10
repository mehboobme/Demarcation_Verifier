#!/usr/bin/env python3
"""
Arch Detection Module for ROSHN Facade Analysis
================================================
Detects curved/arched elements above doors using computer vision.
Used as a secondary validation layer alongside CLIP classifier.

Features:
- Edge detection with Canny
- Contour analysis for curve detection
- Arc/circle detection using Hough Transform
- GPU acceleration with CUDA (if available)
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class ArchDetector:
    """
    Detects architectural arches using computer vision.
    Designed to validate CLIP predictions for traditional vs modern facades.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize arch detector.
        
        Args:
            use_gpu: Use CUDA acceleration if available (RTX 5070)
        """
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.use_gpu:
            logger.info(f"✓ CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) detected")
            logger.info(f"  Device: {cv2.cuda.printCudaDeviceInfo(0)}")
        else:
            logger.info("Using CPU for arch detection")
        
        # Detection parameters (tuned for ROSHN architectural drawings)
        self.canny_low = 50
        self.canny_high = 150
        self.min_arc_radius = 20  # pixels
        self.max_arc_radius = 200  # pixels
        self.curve_threshold = 0.15  # How "curved" a contour must be
    
    def detect_arches(self, image_path: str, debug_output: Optional[str] = None) -> Tuple[bool, float, dict]:
        """
        Detect if image contains door arches.
        
        Args:
            image_path: Path to facade image
            debug_output: Optional path to save debug visualization
        
        Returns:
            (has_arch, confidence, details)
            - has_arch: True if arch detected
            - confidence: 0.0-1.0 confidence score
            - details: Dict with detection metadata
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False, 0.0, {'error': 'Failed to load image'}
            
            # Convert to grayscale
            if self.use_gpu:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                gray = gpu_gray.download()
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for curves
            arch_candidates = []
            for contour in contours:
                if len(contour) < 5:  # Need at least 5 points for ellipse fitting
                    continue
                
                # Fit ellipse to contour
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (MA, ma), angle = ellipse
                    
                    # Filter by size (should be door-sized arch)
                    if self.min_arc_radius <= ma/2 <= self.max_arc_radius:
                        # Check if it's arch-shaped (wider than tall, in upper half)
                        aspect_ratio = MA / ma if ma > 0 else 0
                        if 0.3 <= aspect_ratio <= 3.0:  # Reasonable arch proportions
                            # Calculate curvature score
                            arc_length = cv2.arcLength(contour, True)
                            area = cv2.contourArea(contour)
                            if arc_length > 0:
                                circularity = 4 * np.pi * area / (arc_length ** 2)
                                
                                # Arch should be semi-circular (circularity ~ 0.5-1.0)
                                if circularity > self.curve_threshold:
                                    arch_candidates.append({
                                        'center': (x, y),
                                        'axes': (MA, ma),
                                        'angle': angle,
                                        'circularity': circularity,
                                        'contour': contour
                                    })
                except:
                    continue
            
            # Circle detection using Hough Transform (fallback/validation)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=self.canny_high,
                param2=30,
                minRadius=self.min_arc_radius,
                maxRadius=self.max_arc_radius
            )
            
            circle_count = len(circles[0]) if circles is not None else 0
            
            # Decision logic
            has_arch = len(arch_candidates) > 0 or circle_count > 0
            
            # Calculate confidence based on number and quality of detections
            if arch_candidates:
                # Use highest circularity score
                max_circularity = max(a['circularity'] for a in arch_candidates)
                confidence = min(0.95, 0.5 + max_circularity * 0.45)
            elif circle_count > 0:
                confidence = min(0.85, 0.5 + (circle_count / 10) * 0.35)
            else:
                confidence = 0.0
            
            details = {
                'arch_candidates': len(arch_candidates),
                'circles_detected': circle_count,
                'max_circularity': max(a['circularity'] for a in arch_candidates) if arch_candidates else 0.0,
                'image_size': gray.shape
            }
            
            # Debug visualization
            if debug_output and (arch_candidates or circle_count > 0):
                debug_img = img.copy()
                
                # Draw arch candidates
                for arch in arch_candidates:
                    ellipse_params = (
                        tuple(map(int, arch['center'])),
                        tuple(map(int, arch['axes'])),
                        int(arch['angle'])
                    )
                    cv2.ellipse(debug_img, ellipse_params, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"{arch['circularity']:.2f}", 
                              tuple(map(int, arch['center'])),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw detected circles
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0]:
                        cv2.circle(debug_img, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
                        cv2.circle(debug_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                
                cv2.imwrite(debug_output, debug_img)
                logger.debug(f"Saved arch detection debug image: {debug_output}")
            
            logger.debug(f"Arch detection: {has_arch} (confidence: {confidence:.2%}, "
                        f"candidates: {len(arch_candidates)}, circles: {circle_count})")
            
            return has_arch, confidence, details
        
        except Exception as e:
            logger.error(f"Arch detection failed: {e}")
            return False, 0.0, {'error': str(e)}
    
    def validate_clip_prediction(
        self,
        clip_prediction: str,
        clip_confidence: float,
        image_path: str,
        debug_output: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        Validate CLIP prediction using arch detection.
        
        Args:
            clip_prediction: "T" (traditional) or "M" (modern) from CLIP
            clip_confidence: CLIP confidence score (0.0-1.0)
            image_path: Path to facade image
            debug_output: Optional debug image path
        
        Returns:
            (final_prediction, final_confidence, reason)
        """
        # Run arch detection
        has_arch, arch_confidence, arch_details = self.detect_arches(image_path, debug_output)
        
        # Decision matrix
        if clip_prediction == "T":  # CLIP says Traditional
            if has_arch:
                # Both agree → Traditional with high confidence
                combined_confidence = min(0.99, (clip_confidence + arch_confidence) / 2)
                return "T", combined_confidence, "CLIP + Arch detection both confirm Traditional"
            else:
                # CLIP says Traditional but no arch detected
                # Trust CLIP if confidence is high
                if clip_confidence >= 0.80:
                    return "T", clip_confidence * 0.9, "CLIP confident Traditional (arch detection uncertain)"
                else:
                    # Low CLIP confidence + no arch → Maybe Modern
                    return "M", 0.60, "No arch detected, overriding low-confidence CLIP prediction"
        
        else:  # CLIP says Modern
            if has_arch:
                # CONFLICT: CLIP says Modern but arch detected
                # Arch is strong evidence for Traditional
                if arch_confidence >= 0.70:
                    return "T", arch_confidence, "Arch detected - overriding CLIP prediction to Traditional"
                else:
                    # Weak arch detection, trust CLIP
                    return "M", clip_confidence * 0.9, "CLIP says Modern (arch detection weak)"
            else:
                # Both agree → Modern with high confidence
                combined_confidence = min(0.99, (clip_confidence + (1.0 - arch_confidence)) / 2)
                return "M", combined_confidence, "CLIP + No arch detection confirm Modern"


def test_arch_detector():
    """Test arch detector on sample images."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python arch_detector.py <image_path> [debug_output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detector = ArchDetector(use_gpu=True)
    has_arch, confidence, details = detector.detect_arches(image_path, debug_path)
    
    print(f"\n{'='*60}")
    print(f"Arch Detection Results")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Has Arch: {has_arch}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Details: {details}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_arch_detector()
