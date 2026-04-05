from __future__ import annotations

from pathlib import Path

from core.pipeline import GenerationEngine
from export.export_images import save_image_bundle
from export.export_tables import save_json


def generate_demo_dataset(presets_dir: str | Path, output_dir: str | Path) -> list[Path]:
    engine = GenerationEngine(presets_dir=presets_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for preset_path in engine.list_preset_paths():
        preset_name = preset_path.stem
        result = engine.generate_from_preset(
            preset_path,
            output_size=(1024, 1024),
        )
        if result.view_image is None:
            continue
        preset_out = out / preset_name
        preset_out.mkdir(parents=True, exist_ok=True)

        images = save_image_bundle(
            result.view_image,
            output_dir=preset_out,
            base_name=preset_name,
            formats=("png", "jpg", "tiff"),
        )
        saved.extend(images)

        metadata = result.metadata_for_export()
        metadata["source_preset_file"] = str(preset_path)
        metadata_path = preset_out / f"{preset_name}_metadata.json"
        save_json(metadata, metadata_path)
        saved.append(metadata_path)
    return saved


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    presets = root / "presets"
    output = root / "examples"
    generate_demo_dataset(presets_dir=presets, output_dir=output)

