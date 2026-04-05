from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UIConfig:
    """UI configuration settings"""
    
    # Window settings
    window_min_width: int = 1280
    window_min_height: int = 800
    window_default_width: int = 1400
    window_default_height: int = 900
    
    # Animation settings
    animation_duration_fast: int = 150
    animation_duration_normal: int = 300
    animation_duration_slow: int = 500
    
    # Spacing
    spacing_xs: int = 4
    spacing_sm: int = 8
    spacing_md: int = 12
    spacing_lg: int = 16
    spacing_xl: int = 24
    
    # Border radius
    radius_sm: int = 8
    radius_md: int = 12
    radius_lg: int = 16
    
    # Font sizes
    font_size_xs: int = 11
    font_size_sm: int = 13
    font_size_md: int = 14
    font_size_lg: int = 16
    font_size_xl: int = 18
    font_size_xxl: int = 20
    
    # Preview settings
    preview_max_width: int = 800
    preview_max_height: int = 600
    
    # Performance
    enable_animations: bool = True
    enable_shadows: bool = True
    use_hardware_acceleration: bool = True


@dataclass
class AppPaths:
    """Application paths configuration"""
    
    root: Path
    presets: Path
    profiles: Path
    samples: Path
    exports: Path
    cache: Path
    
    @classmethod
    def from_root(cls, root: Path | str) -> AppPaths:
        """Create paths from root directory"""
        root = Path(root)
        return cls(
            root=root,
            presets=root / "presets_v3",
            profiles=root / "profiles_v3",
            samples=root / "examples" / "factory_v3_output",
            exports=root / "exports",
            cache=root / ".cache",
        )


# Default configuration instance
DEFAULT_UI_CONFIG = UIConfig()
