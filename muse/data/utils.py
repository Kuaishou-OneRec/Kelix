# Predefined aspect ratios for different resolutions
# Reference: Sana/diffusion/data/datasets/utils.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

def get_aspect_ratio_dict(image_size: int) -> Dict[str, tuple]:
    """Get aspect ratio dictionary for given image size.
    
    Args:
        image_size: Base image size (512, 1024, etc.)
        
    Returns:
        Dictionary mapping aspect ratio strings to (height, width) tuples
    """
    if image_size <= 512:
        return ASPECT_RATIO_512
    elif image_size <= 1024:
        return ASPECT_RATIO_1024
    else:
        # Scale up from 1024
        # TODO: support more image sizes
        scale = image_size / 1024
        return {
            k: (int(h * scale), int(w * scale))
            for k, (h, w) in ASPECT_RATIO_1024.items()
        }

def get_closest_ratio(height: int, width: int, aspect_ratios: dict) -> str:
    """Find the closest predefined aspect ratio for given dimensions."""
    ratio = height / width
    return min(aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))

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