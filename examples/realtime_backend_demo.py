from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.virtual_lab_backend import MicroscopeState, VirtualLabBackend


OUT_DIR = Path(__file__).resolve().parent / "realtime_backend_demo_output"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = VirtualLabBackend()
    payload = backend.pipeline.load_preset("steel_tempered_400_textbook")
    payload["resolution"] = [640, 640]
    payload["sample_id"] = "realtime_backend_demo"
    slide = backend.generate_slide(payload)

    states = [
        MicroscopeState(objective=100, stage_x=0.50, stage_y=0.50, output_size=(1024, 1024)),
        MicroscopeState(objective=200, stage_x=0.42, stage_y=0.53, output_size=(1024, 1024)),
        MicroscopeState(objective=400, stage_x=0.42, stage_y=0.53, output_size=(1024, 1024)),
        MicroscopeState(objective=600, stage_x=0.42, stage_y=0.53, output_size=(1024, 1024)),
    ]

    for idx, state in enumerate(states, start=1):
        frame, meta = backend.render_microscope_frame(slide, state)
        out_path = OUT_DIR / f"frame_{idx:02d}_{state.objective}x.png"
        Image.fromarray(frame, mode="L").save(out_path)
        print(f"saved {out_path.name} | focus_quality={meta['focus_quality']:.3f} | level={meta.get('pyramid_level')}")


if __name__ == "__main__":
    main()
