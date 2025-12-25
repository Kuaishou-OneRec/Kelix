# Predefined aspect ratios for different resolutions
# Reference: Sana/diffusion/data/datasets/utils.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

ASPECT_RATIO_256 = {
    '0.25': (128, 512), '0.26': (128, 496), '0.27': (128, 480), '0.28': (128, 464),
    '0.32': (144, 448), '0.33': (144, 432), '0.35': (144, 416), '0.4': (160, 400),
    '0.42': (160, 384), '0.48': (176, 368), '0.5': (176, 352), '0.52': (176, 336),
    '0.57': (192, 336), '0.6': (192, 320), '0.68': (208, 304), '0.72': (208, 288),
    '0.78': (224, 288), '0.82': (224, 272), '0.88': (240, 272), '0.94': (240, 256),
    '1.0': (256, 256),
    '1.07': (256, 240), '1.13': (272, 240), '1.21': (272, 224), '1.29': (288, 224),
    '1.38': (288, 208), '1.46': (304, 208), '1.67': (320, 192), '1.75': (336, 192),
    '2.0': (352, 176), '2.09': (368, 176), '2.4': (384, 160), '2.5': (400, 160),
    '2.89': (416, 144), '3.0': (432, 144), '3.11': (448, 144), '3.62': (464, 128),
    '3.75': (480, 128), '3.88': (496, 128), '4.0': (512, 128),
}

ASPECT_RATIO_512 = {
    '0.25': (256, 1024), '0.26': (256, 992), '0.27': (256, 960), '0.28': (256, 928),
    '0.32': (288, 896), '0.33': (288, 864), '0.35': (288, 832), '0.4': (320, 800),
    '0.42': (320, 768), '0.48': (352, 736), '0.5': (352, 704), '0.52': (352, 672),
    '0.57': (384, 672), '0.6': (384, 640), '0.68': (416, 608), '0.72': (416, 576),
    '0.78': (448, 576), '0.82': (448, 544), '0.88': (480, 544), '0.94': (480, 512),
    '1.0': (512, 512),
    '1.07': (512, 480), '1.13': (544, 480), '1.21': (544, 448), '1.29': (576, 448),
    '1.38': (576, 416), '1.46': (608, 416), '1.67': (640, 384), '1.75': (672, 384),
    '2.0': (704, 352), '2.09': (736, 352), '2.4': (768, 320), '2.5': (800, 320),
    '2.89': (832, 288), '3.0': (864, 288), '3.11': (896, 288), '3.62': (928, 256),
    '3.75': (960, 256), '3.88': (992, 256), '4.0': (1024, 256),
}

# ASPECT_RATIO_768 = {
#     '0.25': (384, 1536), '0.26': (384, 1488), '0.27': (384, 1440), '0.28': (384, 1392),
#     '0.32': (432, 1344), '0.33': (432, 1296), '0.35': (432, 1248), '0.4': (480, 1200),
#     '0.42': (480, 1152), '0.48': (528, 1104), '0.5': (528, 1056), '0.52': (528, 1008),
#     '0.57': (576, 1008), '0.6': (576, 960), '0.68': (624, 912), '0.72': (624, 864),
#     '0.78': (672, 864), '0.82': (672, 816), '0.88': (720, 816), '0.94': (720, 768),
#     '1.0': (768, 768),
#     '1.07': (768, 720), '1.13': (816, 720), '1.21': (816, 672), '1.29': (864, 672),
#     '1.38': (864, 624), '1.46': (912, 624), '1.67': (960, 576), '1.75': (1008, 576),
#     '2.0': (1056, 528), '2.09': (1104, 528), '2.4': (1152, 480), '2.5': (1200, 480),
#     '2.89': (1248, 432), '3.0': (1296, 432), '3.11': (1344, 432), '3.62': (1392, 384),
#     '3.75': (1440, 384), '3.88': (1488, 384), '4.0': (1536, 384),
# }

ASPECT_RATIO_1024 = {
    '0.25': (512, 2048), '0.26': (512, 1984), '0.27': (512, 1920), '0.28': (512, 1856),
    '0.32': (576, 1792), '0.33': (576, 1728), '0.35': (576, 1664), '0.4': (640, 1600),
    '0.42': (640, 1536), '0.48': (704, 1472), '0.5': (704, 1408), '0.52': (704, 1344),
    '0.57': (768, 1344), '0.6': (768, 1280), '0.68': (832, 1216), '0.72': (832, 1152),
    '0.78': (896, 1152), '0.82': (896, 1088), '0.88': (960, 1088), '0.94': (960, 1024),
    '1.0': (1024, 1024),
    '1.07': (1024, 960), '1.13': (1088, 960), '1.21': (1088, 896), '1.29': (1152, 896),
    '1.38': (1152, 832), '1.46': (1216, 832), '1.67': (1280, 768), '1.75': (1344, 768),
    '2.0': (1408, 704), '2.09': (1472, 704), '2.4': (1536, 640), '2.5': (1600, 640),
    '2.89': (1664, 576), '3.0': (1728, 576), '3.11': (1792, 576), '3.62': (1856, 512),
    '3.75': (1920, 512), '3.88': (1984, 512), '4.0': (2048, 512),
}

ASPECT_RATIO_1536 = {
    '0.25': (768, 3072), '0.26': (768, 2976), '0.27': (768, 2880), '0.28': (768, 2784),
    '0.32': (864, 2688), '0.33': (864, 2592), '0.35': (864, 2496), '0.4': (960, 2400),
    '0.42': (960, 2304), '0.48': (1056, 2208), '0.5': (1056, 2112), '0.52': (1056, 2016),
    '0.57': (1152, 2016), '0.6': (1152, 1920), '0.68': (1248, 1824), '0.72': (1248, 1728),
    '0.78': (1344, 1728), '0.82': (1344, 1632), '0.88': (1440, 1632), '0.94': (1440, 1536),
    '1.0': (1536, 1536),
    '1.07': (1536, 1440), '1.13': (1632, 1440), '1.21': (1632, 1344), '1.29': (1728, 1344),
    '1.38': (1728, 1248), '1.46': (1824, 1248), '1.67': (1920, 1152), '1.75': (2016, 1152),
    '2.0': (2112, 1056), '2.09': (2208, 1056), '2.4': (2304, 960), '2.5': (2400, 960),
    '2.89': (2496, 864), '3.0': (2592, 864), '3.11': (2688, 864), '3.62': (2784, 768),
    '3.75': (2880, 768), '3.88': (2976, 768), '4.0': (3072, 768),
}

ASPECT_RATIO_2048 = {
    '0.25': (1024, 4096), '0.26': (1024, 3968), '0.27': (1024, 3840), '0.28': (1024, 3712),
    '0.32': (1152, 3584), '0.33': (1152, 3456), '0.35': (1152, 3328), '0.4': (1280, 3200),
    '0.42': (1280, 3072), '0.48': (1408, 2944), '0.5': (1408, 2816), '0.52': (1408, 2688),
    '0.57': (1536, 2688), '0.6': (1536, 2560), '0.68': (1664, 2432), '0.72': (1664, 2304),
    '0.78': (1792, 2304), '0.82': (1792, 2176), '0.88': (1920, 2176), '0.94': (1920, 2048),
    '1.0': (2048, 2048),
    '1.07': (2048, 1920), '1.13': (2176, 1920), '1.21': (2176, 1792), '1.29': (2304, 1792),
    '1.38': (2304, 1664), '1.46': (2432, 1664), '1.67': (2560, 1536), '1.75': (2688, 1536),
    '2.0': (2816, 1408), '2.09': (2944, 1408), '2.4': (3072, 1280), '2.5': (3200, 1280),
    '2.89': (3328, 1152), '3.0': (3456, 1152), '3.11': (3584, 1152), '3.62': (3712, 1024),
    '3.75': (3840, 1024), '3.88': (3968, 1024), '4.0': (4096, 1024),
}

def get_aspect_ratio_dict(image_size: int) -> Dict[str, tuple]:
    """Get aspect ratio dictionary for given image size.
    
    Args:
        image_size: Base image size (256, 512, 768, 1024, 1536, 2048, etc.)
        
    Returns:
        Dictionary mapping aspect ratio strings to (height, width) tuples
    """
    if image_size <= 256:
        return ASPECT_RATIO_256
    elif image_size <= 512:
        return ASPECT_RATIO_512
    # elif image_size <= 768:
    #     return ASPECT_RATIO_768
    elif image_size <= 1024:
        return ASPECT_RATIO_1024
    elif image_size <= 1536:
        return ASPECT_RATIO_1536
    elif image_size <= 2048:
        return ASPECT_RATIO_2048
    else:
        # Scale up from 2048 for larger sizes
        scale = image_size / 2048
        return {
            k: (int(h * scale), int(w * scale))
            for k, (h, w) in ASPECT_RATIO_2048.items()
        }

def get_closest_ratio(height: int, width: int, aspect_ratios: dict) -> str:
    """Find the closest predefined aspect ratio for given dimensions."""
    ratio = height / width
    return min(aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))


# Standard resolution levels for multi-scale training
# RESOLUTION_LEVELS = [256, 512, 768, 1024, 1536, 2048]

RESOLUTION_LEVELS = [256, 512, 1024, 1536, 2048]

def get_resolution_level(
    height: int,
    width: int,
    resolution_levels: List[int] = None,
) -> int:
    """Determine the resolution level for an image based on its dimensions.
    
    Uses the geometric mean (sqrt(height * width)) to classify images into
    predefined resolution buckets. The returned resolution level is always
    less than or equal to the image's effective resolution to avoid upsampling.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        resolution_levels: List of resolution levels to choose from.
                          Defaults to RESOLUTION_LEVELS.
    
    Returns:
        The largest resolution level that doesn't exceed the image's effective
        resolution, or the smallest available level if image is too small.
        
    Example:
        >>> get_resolution_level(512, 512)   # Returns 512
        >>> get_resolution_level(1024, 1024) # Returns 1024
        >>> get_resolution_level(768, 1280)  # Returns 768 (sqrt(768*1280) ≈ 992, largest <= 992)
        >>> get_resolution_level(256, 1024)  # Returns 512 (sqrt(256*1024) = 512)
        >>> get_resolution_level(200, 200)   # Returns 256 (smallest available, even though image is smaller)
    """
    if resolution_levels is None:
        resolution_levels = RESOLUTION_LEVELS
    
    # Use geometric mean as the effective resolution
    effective_res = (height * width) ** 0.5
    
    # Filter to only resolution levels <= effective_res (avoid upsampling)
    valid_levels = [r for r in resolution_levels if r <= effective_res]
    
    if valid_levels:
        # Return the largest valid level (closest to effective_res without exceeding)
        return max(valid_levels)
    else:
        # Image is smaller than all levels, return the smallest level
        return min(resolution_levels)

def get_closest_size(
    height: int,
    width: int,
    aspect_ratios: Dict[str, Tuple[int, int]],
) -> Tuple[int, int]:
    """Transform image to target resolution with closest aspect ratio.
    
    Args:
        image: PIL Image (already RGB).
        resolution: Target resolution budget.
        aspect_ratios: Aspect ratio dict for this resolution.
        
    Returns:
        Tuple of (transformed_tensor, aspect_ratio_key, target_size).
    """
    closest_ratio = get_closest_ratio(height, width, aspect_ratios)
    target_h, target_w = aspect_ratios[closest_ratio]
    
    return target_h, target_w
# =============================================================================
# Resolution Budget Configuration for Dynamic Multi-Scale Training
# =============================================================================

@dataclass
class ResolutionBudget:
    """Single resolution budget entry."""
    size: int           # e.g. 512, 768, 1024
    batch_size: int     # batch size for this resolution


@dataclass
class ResolutionBudgetConfig:
    """Complete resolution budget configuration with weight scheduling.
    
    Supports curriculum learning where weights interpolate linearly from
    start_weights to end_weights based on training progress.
    
    Args:
        budgets: List of ResolutionBudget entries
        start_weights: Weights at training start (low-res heavy)
        end_weights: Weights at training end (high-res heavy)
        
    Example:
        >>> config = ResolutionBudgetConfig(
        ...     budgets=[ResolutionBudget(512, 32), ResolutionBudget(1024, 8)],
        ...     start_weights=[0.8, 0.2],  # 80% 512, 20% 1024 at start
        ...     end_weights=[0.2, 0.8],    # 20% 512, 80% 1024 at end
        ... )
        >>> config.get_weights(0.0)   # [0.8, 0.2]
        >>> config.get_weights(0.5)   # [0.5, 0.5]
        >>> config.get_weights(1.0)   # [0.2, 0.8]
    """
    budgets: List[ResolutionBudget]
    start_weights: List[float]  # weights at training start (low-res heavy)
    end_weights: List[float]    # weights at training end (high-res heavy)
    
    def __post_init__(self):
        n = len(self.budgets)
        assert len(self.start_weights) == n, \
            f"start_weights length {len(self.start_weights)} != budgets {n}"
        assert len(self.end_weights) == n, \
            f"end_weights length {len(self.end_weights)} != budgets {n}"
        # Normalize weights
        self.start_weights = self._normalize(self.start_weights)
        self.end_weights = self._normalize(self.end_weights)
    
    @staticmethod
    def _normalize(weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1."""
        total = sum(weights)
        if total == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]
    
    def get_weights(self, progress: float) -> List[float]:
        """Get interpolated weights based on training progress.
        
        Args:
            progress: Training progress in [0, 1] (current_step / total_steps)
            
        Returns:
            List of weights, one per resolution budget
        """
        progress = max(0.0, min(1.0, progress))  # clamp to [0, 1]
        weights = [
            start + progress * (end - start)
            for start, end in zip(self.start_weights, self.end_weights)
        ]
        return self._normalize(weights)


# Default config: low-res heavy early, high-res heavy late
DEFAULT_RESOLUTION_BUDGETS = ResolutionBudgetConfig(
    budgets=[
        ResolutionBudget(size=512, batch_size=32),
        ResolutionBudget(size=768, batch_size=16),
        ResolutionBudget(size=1024, batch_size=8),
    ],
    start_weights=[0.7, 0.2, 0.1],  # 70% 512, 20% 768, 10% 1024 at start
    end_weights=[0.1, 0.2, 0.7],    # 10% 512, 20% 768, 70% 1024 at end
)


def parse_resolution_budgets(
    budgets_str: str,
    start_weights_str: Optional[str] = None,
    end_weights_str: Optional[str] = None,
) -> ResolutionBudgetConfig:
    """Parse resolution budgets from CLI strings.
    
    Args:
        budgets_str: Format "512:32,768:16,1024:8" (size:batch_size)
        start_weights_str: Format "0.7,0.2,0.1" (weights at start)
        end_weights_str: Format "0.1,0.2,0.7" (weights at end)
        
    Returns:
        ResolutionBudgetConfig with parsed values
        
    Example:
        >>> config = parse_resolution_budgets(
        ...     "512:32,768:16,1024:8",
        ...     "0.7,0.2,0.1",
        ...     "0.1,0.2,0.7"
        ... )
    """
    budgets = []
    for entry in budgets_str.split(","):
        parts = entry.strip().split(":")
        size = int(parts[0])
        batch_size = int(parts[1])
        budgets.append(ResolutionBudget(size=size, batch_size=batch_size))
    
    n = len(budgets)
    
    # Parse or default start weights
    if start_weights_str:
        start_weights = [float(w.strip()) for w in start_weights_str.split(",")]
    else:
        # Default: favor low-res (first budget gets 70%)
        if n > 1:
            start_weights = [0.7] + [0.3 / (n - 1)] * (n - 1)
        else:
            start_weights = [1.0]
    
    # Parse or default end weights
    if end_weights_str:
        end_weights = [float(w.strip()) for w in end_weights_str.split(",")]
    else:
        # Default: favor high-res (last budget gets 70%)
        if n > 1:
            end_weights = [0.3 / (n - 1)] * (n - 1) + [0.7]
        else:
            end_weights = [1.0]
    
    return ResolutionBudgetConfig(
        budgets=budgets,
        start_weights=start_weights,
        end_weights=end_weights,
    )