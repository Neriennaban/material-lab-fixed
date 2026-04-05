"""Version information for the application"""

from __future__ import annotations

__version__ = "4.0.0"
__version_info__ = (4, 0, 0)

VERSION_MAJOR = 4
VERSION_MINOR = 0
VERSION_PATCH = 0

VERSION_STRING = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

# Release information
RELEASE_DATE = "2026-03-07"
RELEASE_NAME = "Modern UI & Performance"

# Feature flags
FEATURES = {
    "modern_ui": True,
    "animations": True,
    "notifications": True,
    "advanced_cache": True,
    "performance_monitoring": True,
    "validation": True,
    "enhanced_logging": True,
}


def get_version() -> str:
    """Get version string"""
    return VERSION_STRING


def get_version_info() -> dict:
    """Get detailed version information"""
    return {
        "version": VERSION_STRING,
        "version_info": __version_info__,
        "release_date": RELEASE_DATE,
        "release_name": RELEASE_NAME,
        "features": FEATURES,
    }


def print_version() -> None:
    """Print version information"""
    print(f"Metallography Lab v{VERSION_STRING}")
    print(f"Release: {RELEASE_NAME}")
    print(f"Date: {RELEASE_DATE}")
    print("\nEnabled features:")
    for feature, enabled in FEATURES.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")


if __name__ == "__main__":
    print_version()
