"""
Image Scoring and Filtering Module
Provides basic image quality scoring and filtering functionality
"""

from abc import ABC, abstractmethod
from PIL import Image, ImageStat, ImageEnhance
import math
import os
import uuid
from collections import defaultdict
from typing import Dict, Tuple, Optional
import shutil


class BaseScorer(ABC):
    """Base scorer abstract class"""
    
    @abstractmethod
    def score(self, image: Image.Image) -> float:
        """
        Score an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Score value, normalized to 0-1 range
        """
        pass


class BrightnessScorer(BaseScorer):
    """Brightness scorer"""
    
    def score(self, image: Image.Image) -> float:
        """
        Calculate image brightness score
        
        Returns:
            Brightness score (0-1), higher value means brighter
        """
        # Convert to grayscale for brightness calculation
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
            
        stat = ImageStat.Stat(gray_image)
        brightness = stat.mean[0] / 255.0  # Normalize to 0-1
        
        return float(brightness)


class SaturationScorer(BaseScorer):
    """Saturation scorer"""
    
    def score(self, image: Image.Image) -> float:
        """
        Calculate image saturation score
        
        Returns:
            Saturation score (0-1), higher value means more saturated
        """
        if image.mode != 'HSV':
            hsv_image = image.convert('HSV')
        else:
            hsv_image = image
            
        # Extract saturation channel
        _, saturation, _ = hsv_image.split()
        stat = ImageStat.Stat(saturation)
        saturation_score = stat.mean[0] / 255.0  # Normalize to 0-1
        
        return float(saturation_score)


class ContrastScorer(BaseScorer):
    """Contrast scorer"""
    
    def score(self, image: Image.Image) -> float:
        """
        Calculate image contrast score
        
        Returns:
            Contrast score (0-1), higher value means higher contrast
        """
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
            
        stat = ImageStat.Stat(gray_image)
        
        # Use standard deviation as contrast measure
        if len(stat.stddev) > 0:
            std_dev = stat.stddev[0]
        else:
            std_dev = 0
            
        # Normalize to 0-1 (assuming max standard deviation is 128, empirical value)
        contrast = min(std_dev / 128.0, 1.0)
        
        return float(contrast)


class ColorfulnessScorer(BaseScorer):
    """Colorfulness scorer"""
    
    def score(self, image: Image.Image) -> float:
        """
        Calculate image colorfulness score
        
        Returns:
            Colorfulness score (0-1), higher value means more colorful
        """
        if image.mode != 'RGB':
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
            
        # Split image into R, G, B channels
        r, g, b = rgb_image.split()
        
        # Calculate mean and standard deviation for each channel
        r_stat = ImageStat.Stat(r)
        g_stat = ImageStat.Stat(g)
        b_stat = ImageStat.Stat(b)
        
        r_mean = r_stat.mean[0]
        g_mean = g_stat.mean[0]
        b_mean = b_stat.mean[0]
        
        r_std = r_stat.stddev[0] if len(r_stat.stddev) > 0 else 0
        g_std = g_stat.stddev[0] if len(g_stat.stddev) > 0 else 0
        b_std = b_stat.stddev[0] if len(b_stat.stddev) > 0 else 0
        
        # Calculate colorfulness formula
        rg_diff = r_mean - g_mean
        yb_diff = (r_mean + g_mean) / 2 - b_mean
        
        std_root = math.sqrt(r_std**2 + g_std**2 + b_std**2)
        mean_root = math.sqrt(rg_diff**2 + yb_diff**2)
        
        colorfulness = std_root + 0.3 * mean_root
        
        # Normalize to 0-1 (empirical value, max around 150)
        normalized = min(colorfulness / 150.0, 1.0)
        
        return float(normalized)


class BaseFilter(ABC):
    """Base filter abstract class"""
    
    @abstractmethod
    def filter(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Filter an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple[bool, str]: 
                - bool: Whether the image passes the filter (True means pass)
                - str: Reason for rejection (empty string if passed)
        """
        pass


class ImageQualityFilter(BaseFilter):
    """Image quality filter with statistics and filtered cases storage"""
    
    # Define rejection reason constants
    REASON_TOO_DARK = "too_dark"
    REASON_TOO_BRIGHT = "too_bright"
    REASON_UNDERSATURATED = "undersaturated"
    REASON_OVERSATURATED = "oversaturated"
    REASON_LOW_CONTRAST = "low_contrast"
    REASON_HIGH_CONTRAST = "high_contrast"
    REASON_LOW_COLORFULNESS = "low_colorfulness"
    
    def __init__(self, 
                 brightness_threshold: Tuple[float, float] = (0.15, 0.85),
                 saturation_threshold: Tuple[float, float] = (0.05, 0.9),
                 contrast_threshold: Tuple[float, float] = (0.1, 0.9),
                 colorfulness_threshold: float = 0.05,
                 filtered_cases_dir: Optional[str] = None,
                 max_cases_per_reason: int = 20):
        """
        Initialize quality filter
        
        Args:
            brightness_threshold: Brightness threshold (min, max), below min is too dark, above max is too bright
            saturation_threshold: Saturation threshold (min, max), below min is undersaturated, above max is oversaturated
            contrast_threshold: Contrast threshold (min, max), below min is low contrast, above max is high contrast
            colorfulness_threshold: Minimum colorfulness threshold, below this is considered low colorfulness
            filtered_cases_dir: Directory to save filtered images (None means don't save)
            max_cases_per_reason: Maximum number of images to save per rejection reason
        """
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
        self.contrast_threshold = contrast_threshold
        self.colorfulness_threshold = colorfulness_threshold
        self.filtered_cases_dir = filtered_cases_dir
        self.max_cases_per_reason = max_cases_per_reason
        
        # Initialize scorers
        self.brightness_scorer = BrightnessScorer()
        self.saturation_scorer = SaturationScorer()
        self.contrast_scorer = ContrastScorer()
        self.colorfulness_scorer = ColorfulnessScorer()
        
        # Initialize statistics
        self._reset_statistics()
        
        # Initialize filtered cases storage
        self._init_filtered_cases_storage()
    
    def _reset_statistics(self):
        """Reset all statistics counters"""
        self.total_processed = 0
        self.passed_count = 0
        self.rejected_count = 0
        self.rejection_reasons = defaultdict(int)
        self.combined_rejection_reasons = defaultdict(int)
        
        # Track saved cases per reason
        self.saved_cases_per_reason = defaultdict(int)
    
    def _init_filtered_cases_storage(self):
        """Initialize directories for saving filtered cases"""
        if self.filtered_cases_dir is None:
            return
            
        # Create main directory if it doesn't exist
        os.makedirs(self.filtered_cases_dir, exist_ok=True)
        
        # Create subdirectories for each rejection reason
        reason_dirs = [
            self.REASON_TOO_DARK,
            self.REASON_TOO_BRIGHT,
            self.REASON_UNDERSATURATED,
            self.REASON_OVERSATURATED,
            self.REASON_LOW_CONTRAST,
            self.REASON_HIGH_CONTRAST,
            self.REASON_LOW_COLORFULNESS,
            "multiple_reasons"  # For images with multiple rejection reasons
        ]
        
        for reason_dir in reason_dirs:
            dir_path = os.path.join(self.filtered_cases_dir, reason_dir)
            os.makedirs(dir_path, exist_ok=True)
            
            # Count existing files in each directory
            if os.path.exists(dir_path):
                jpg_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
                self.saved_cases_per_reason[reason_dir] = len(jpg_files)
    
    def _save_filtered_image(self, image: Image.Image, reasons: list) -> bool:
        """
        Save filtered image to appropriate directory
        
        Args:
            image: Image to save
            reasons: List of rejection reasons
            
        Returns:
            True if image was saved, False otherwise
        """
        if self.filtered_cases_dir is None:
            return False
            
        # Determine the appropriate directory
        if len(reasons) == 1:
            # Single reason - use the reason directory
            reason_const = self._get_reason_constant(reasons[0])
            save_dir = os.path.join(self.filtered_cases_dir, reason_const)
        else:
            # Multiple reasons - use the combined directory
            save_dir = os.path.join(self.filtered_cases_dir, "multiple_reasons")
            reason_const = "multiple_reasons"
        
        # Check if we've reached the maximum for this reason
        if self.saved_cases_per_reason[reason_const] >= self.max_cases_per_reason:
            return False
        
        # Generate unique filename
        filename = f"{uuid.uuid4().hex[:8]}_{reason_const}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Ensure the image is in RGB mode for saving as JPEG
        if image.mode != 'RGB':
            save_image = image.convert('RGB')
        else:
            save_image = image
        
        try:
            # Save the image
            save_image.save(filepath, 'JPEG', quality=85)
            self.saved_cases_per_reason[reason_const] += 1
            
            # Also save a text file with details
            details_file = os.path.splitext(filepath)[0] + ".txt"
            with open(details_file, 'w') as f:
                f.write(f"Rejection reasons: {'; '.join(reasons)}\n")
                f.write(f"Image mode: {image.mode}\n")
                f.write(f"Image size: {image.size}\n")
                
            return True
        except Exception as e:
            print(f"Failed to save filtered image: {e}")
            return False
    
    def _get_reason_constant(self, reason_str: str) -> str:
        """
        Extract the reason constant from a reason string
        
        Args:
            reason_str: Reason string from filter method
            
        Returns:
            Reason constant
        """
        if "too dark" in reason_str:
            return self.REASON_TOO_DARK
        elif "too bright" in reason_str:
            return self.REASON_TOO_BRIGHT
        elif "undersaturated" in reason_str:
            return self.REASON_UNDERSATURATED
        elif "oversaturated" in reason_str:
            return self.REASON_OVERSATURATED
        elif "low contrast" in reason_str:
            return self.REASON_LOW_CONTRAST
        elif "high contrast" in reason_str:
            return self.REASON_HIGH_CONTRAST
        elif "low colorfulness" in reason_str:
            return self.REASON_LOW_COLORFULNESS
        else:
            return "unknown"
    
    def filter(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Filter image, check brightness, saturation, contrast and colorfulness
        
        Returns:
            Tuple[bool, str]: Whether passed and reason
        """
        self.total_processed += 1
        rejection_reasons = []
        
        # Check brightness
        brightness = self.brightness_scorer.score(image)
        if brightness < self.brightness_threshold[0]:
            reason = f"too dark (brightness: {brightness:.3f} < {self.brightness_threshold[0]})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_TOO_DARK] += 1
        elif brightness > self.brightness_threshold[1]:
            reason = f"too bright (brightness: {brightness:.3f} > {self.brightness_threshold[1]})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_TOO_BRIGHT] += 1
        
        # Check saturation
        saturation = self.saturation_scorer.score(image)
        if saturation < self.saturation_threshold[0]:
            reason = f"undersaturated (saturation: {saturation:.3f} < {self.saturation_threshold[0]})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_UNDERSATURATED] += 1
        elif saturation > self.saturation_threshold[1]:
            reason = f"oversaturated (saturation: {saturation:.3f} > {self.saturation_threshold[1]})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_OVERSATURATED] += 1
        
        # Check contrast
        contrast = self.contrast_scorer.score(image)
        if contrast < self.contrast_threshold[0]:
            reason = f"low contrast (contrast: {contrast:.3f} < {self.contrast_threshold[0]})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_LOW_CONTRAST] += 1
        elif contrast > self.contrast_threshold[1]:
            reason = f"high contrast (contrast: {contrast:.3f} > {self.contrast_threshold[1]})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_HIGH_CONTRAST] += 1
        
        # Check colorfulness
        colorfulness = self.colorfulness_scorer.score(image)
        if colorfulness < self.colorfulness_threshold:
            reason = f"low colorfulness (colorfulness: {colorfulness:.3f} < {self.colorfulness_threshold})"
            rejection_reasons.append(reason)
            self.rejection_reasons[self.REASON_LOW_COLORFULNESS] += 1
        
        # Aggregate results
        if rejection_reasons:
            self.rejected_count += 1
            
            # Save filtered image if configured
            if self.filtered_cases_dir is not None:
                self._save_filtered_image(image, rejection_reasons)
            
            # Record combined reasons (for images with multiple issues)
            combined_key = "+".join(sorted([r.split()[0] for r in rejection_reasons]))
            self.combined_rejection_reasons[combined_key] += 1
            
            return False, "; ".join(rejection_reasons)
        else:
            self.passed_count += 1
            return True, ""
    
    def get_statistics(self) -> Dict:
        """
        Get filter statistics
        
        Returns:
            Dictionary containing filter statistics
        """

        stats = {
            "total_processed": self.total_processed,
            "passed_count": self.passed_count,
            "passed_rate": self.passed_count / max(self.total_processed, 1),
            "rejected_count": self.rejected_count,
            "rejected_rate": self.rejected_count / max(self.total_processed, 1),
            "rejection_reasons": dict(self.rejection_reasons),
            "combined_rejection_reasons": dict(self.combined_rejection_reasons),
            "saved_cases_per_reason": dict(self.saved_cases_per_reason),
            "max_cases_per_reason": self.max_cases_per_reason
        }
        
        # Add information about filtered cases storage
        if self.filtered_cases_dir:
            stats["filtered_cases_dir"] = self.filtered_cases_dir
            stats["storage_active"] = True
            # Check if any reason has reached max capacity
            full_reasons = []
            for reason, count in self.saved_cases_per_reason.items():
                if count >= self.max_cases_per_reason:
                    full_reasons.append(reason)
            stats["full_reasons"] = full_reasons
        else:
            stats["filtered_cases_dir"] = None
            stats["storage_active"] = False
            
        return stats
    
    def print_statistics(self, detailed: bool = False):
        """
        Print filter statistics
        
        Args:
            detailed: Whether to print detailed statistics including combined rejection reasons
        """
        stats = self.get_statistics()
        
        print("=" * 60)
        print("IMAGE QUALITY FILTER STATISTICS")
        print("=" * 60)
        print(f"Total processed: {stats['total_processed']}")
        print(f"Passed: {stats['passed_count']} ({stats['passed_rate']*100:.1f}%)")
        print(f"Rejected: {stats['rejected_count']} ({stats['rejected_rate']*100:.1f}%)")
        
        # Print storage information if active
        if stats['storage_active']:
            print(f"\nFiltered cases storage: {stats['filtered_cases_dir']}")
            print(f"Max cases per reason: {stats['max_cases_per_reason']}")
            print("\nSaved cases per reason:")
            for reason, count in sorted(stats['saved_cases_per_reason'].items()):
                status = "FULL" if count >= self.max_cases_per_reason else f"{count}/{self.max_cases_per_reason}"
                print(f"  - {reason}: {status}")
            
            # Print full reasons if any
            if stats['full_reasons']:
                print(f"\nReasons at capacity: {', '.join(stats['full_reasons'])}")
        
        if stats['rejected_count'] > 0:
            print("\nRejection reasons (individual):")
            for reason, count in sorted(stats['rejection_reasons'].items()):
                percentage = count / max(stats['rejected_count'], 1) * 100
                print(f"  - {reason}: {count} ({percentage:.1f}% of rejected)")
            
            if detailed and stats['combined_rejection_reasons']:
                print("\nCombined rejection reasons:")
                for combined_reason, count in sorted(stats['combined_rejection_reasons'].items()):
                    percentage = count / max(stats['rejected_count'], 1) * 100
                    print(f"  - {combined_reason}: {count} ({percentage:.1f}% of rejected)")
        print("=" * 60)
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self._reset_statistics()
        print("Statistics have been reset.")
    
    def clear_filtered_cases(self, confirm: bool = True):
        """
        Clear all saved filtered cases
        
        Args:
            confirm: If True, ask for confirmation before deleting
        """
        if self.filtered_cases_dir is None:
            print("No filtered cases directory configured.")
            return
            
        if confirm:
            response = input(f"Are you sure you want to delete all files in {self.filtered_cases_dir}? [y/N]: ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return
        
        try:
            if os.path.exists(self.filtered_cases_dir):
                shutil.rmtree(self.filtered_cases_dir)
                print(f"Deleted directory: {self.filtered_cases_dir}")
                # Reinitialize the directory structure
                self._init_filtered_cases_storage()
            else:
                print(f"Directory does not exist: {self.filtered_cases_dir}")
        except Exception as e:
            print(f"Failed to clear filtered cases: {e}")
    
    def get_storage_info(self) -> Dict:
        """
        Get detailed information about filtered cases storage
        
        Returns:
            Dictionary with storage information
        """
        if self.filtered_cases_dir is None:
            return {"active": False}
        
        info = {
            "active": True,
            "directory": self.filtered_cases_dir,
            "max_cases_per_reason": self.max_cases_per_reason,
            "reasons": {}
        }
        
        # Check each reason directory
        reason_dirs = [
            self.REASON_TOO_DARK,
            self.REASON_TOO_BRIGHT,
            self.REASON_UNDERSATURATED,
            self.REASON_OVERSATURATED,
            self.REASON_LOW_CONTRAST,
            self.REASON_HIGH_CONTRAST,
            self.REASON_LOW_COLORFULNESS,
            "multiple_reasons"
        ]
        
        for reason in reason_dirs:
            dir_path = os.path.join(self.filtered_cases_dir, reason)
            if os.path.exists(dir_path):
                jpg_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
                txt_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.txt')]
                info["reasons"][reason] = {
                    "jpg_count": len(jpg_files),
                    "txt_count": len(txt_files),
                    "at_capacity": len(jpg_files) >= self.max_cases_per_reason
                }
            else:
                info["reasons"][reason] = {"jpg_count": 0, "txt_count": 0, "at_capacity": False}
        
        return info


# Example usage
if __name__ == "__main__":
    import tempfile
    import os
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="filter_test_")
    print(f"Test directory: {test_dir}")
    
    # Example: Create filter instance with filtered cases storage
    filter = ImageQualityFilter(
        filtered_cases_dir=test_dir,
        max_cases_per_reason=5  # Small limit for testing
    )
    
    # Example: Create test images (using solid color images for simulation)
    test_images = [
        ("dark_image", Image.new('RGB', (100, 100), (10, 10, 10))),      # Dark image
        ("bright_image", Image.new('RGB', (100, 100), (250, 250, 250))), # Bright image
        ("gray_image", Image.new('RGB', (100, 100), (128, 128, 128))),   # Neutral gray image
        ("red_image", Image.new('RGB', (100, 100), (255, 0, 0))),        # High saturation red
        ("dark_image2", Image.new('RGB', (100, 100), (5, 5, 5))),        # Another dark image
        ("bright_image2", Image.new('RGB', (100, 100), (245, 245, 245))),# Another bright image
        ("low_sat", Image.new('RGB', (100, 100), (200, 200, 200))),      # Low saturation
        ("high_sat", Image.new('RGB', (100, 100), (255, 0, 255))),       # High saturation
    ]
    
    print("\nTesting images with filtered cases storage:")
    for name, img in test_images:
        passed, reason = filter.filter(img)
        status = "PASSED" if passed else "REJECTED"
        print(f"{name}: {status} - {reason}")
    
    # Print statistics
    print("\n")
    filter.print_statistics(detailed=True)
    
    # Print storage information
    storage_info = filter.get_storage_info()
    if storage_info["active"]:
        print("\nFiltered Cases Storage Info:")
        for reason, info in storage_info["reasons"].items():
            if info["jpg_count"] > 0:
                print(f"  {reason}: {info['jpg_count']} images, capacity: {'FULL' if info['at_capacity'] else 'available'}")
    
    # Test with image that has multiple rejection reasons
    print("\nTesting image with multiple rejection reasons:")
    multi_issue_img = Image.new('RGB', (100, 100), (5, 5, 5))  # Dark and low contrast
    passed, reason = filter.filter(multi_issue_img)
    print(f"Multi-issue image: {'PASSED' if passed else 'REJECTED'} - {reason}")
    
    # Test clearing filtered cases (commented out for safety)
    # filter.clear_filtered_cases(confirm=False)
    
    # Clean up test directory
    print(f"\nTest directory contents preserved at: {test_dir}")
    print("(Uncomment the clear_filtered_cases line to clean up automatically)")