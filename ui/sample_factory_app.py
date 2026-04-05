from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from core.crm_fe_c_generator import format_fraction_summary, generate_crm_fe_c_rgb
from core.generator_phase_map import supported_stages
from core.materials import MaterialPreset
from core.phase_diagrams import infer_system_from_context, render_detailed_diagram
from core.pipeline import GenerationEngine, GenerationResult
from export.export_images import save_image
from export.export_tables import save_json, save_measurements_csv


PHASE_SYSTEMS = ["fe-c", "fe-si", "al-si", "cu-zn", "al-cu-mg"]
COOLING_MODES = ["equilibrium", "slow_cool", "quenched", "tempered", "aged", "cold_worked", "natural_aged"]
IRON_TYPES = ["auto", "white_cast_iron", "gray_cast_iron"]


@dataclass(slots=True)
class FactorySample:
    image: np.ndarray
    metadata: dict[str, Any]
    sample_metadata: dict[str, Any]
    mode: str
    source_name: str


class SampleFactoryApp:
    """Separate software for research sample generation."""

    def __init__(self, root: tk.Tk, presets_dir: str | Path | None = None) -> None:
        self.root = root
        self.root.title("Генератор образцов")
        self.root.geometry("1600x940")
        self.root.minsize(1300, 780)

        self.engine = GenerationEngine(presets_dir=presets_dir)
        self.preview_sample: FactorySample | None = None
        self.preview_engine_result: GenerationResult | None = None
        self.tk_preview: ImageTk.PhotoImage | None = None
        self.tk_diagram: ImageTk.PhotoImage | None = None

        self._build_layout()
        self._load_preset_names()

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, padding=10)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=0)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        left = ttk.Frame(shell, width=560)
        right = ttk.Frame(shell)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_controls(left)
        self._build_preview(right)
        self._build_diagram(right)

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        mode_box = ttk.LabelFrame(parent, text="Режим", padding=8)
        mode_box.grid(row=0, column=0, sticky="ew")
        self.mode_var = tk.StringVar(value="crm_fe_c")
        ttk.Radiobutton(mode_box, text="CRM Fe-C (как в примере)", value="crm_fe_c", variable=self.mode_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(mode_box, text="По пресету", value="preset", variable=self.mode_var).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Radiobutton(mode_box, text="Конструктор фаз", value="phase", variable=self.mode_var).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )

        preset_box = ttk.LabelFrame(parent, text="Источник пресета", padding=8)
        preset_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        preset_box.columnconfigure(1, weight=1)
        ttk.Label(preset_box, text="Пресет").grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(preset_box, textvariable=self.preset_var, state="readonly")
        self.preset_combo.grid(row=0, column=1, sticky="ew")

        crm_box = ttk.LabelFrame(parent, text="Параметры CRM Fe-C", padding=8)
        crm_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        crm_box.columnconfigure(1, weight=1)

        self.crm_carbon_var = tk.DoubleVar(value=0.45)
        self.crm_iron_type_var = tk.StringVar(value="auto")
        self.crm_grains_var = tk.IntVar(value=130)
        self.crm_distortion_var = tk.DoubleVar(value=0.65)
        self.crm_temp_var = tk.DoubleVar(value=20.0)

        ttk.Label(crm_box, text="Углерод, мас.%").grid(row=0, column=0, sticky="w")
        ttk.Entry(crm_box, textvariable=self.crm_carbon_var).grid(row=0, column=1, sticky="ew")
        ttk.Label(crm_box, text="Тип чугуна/стали").grid(row=1, column=0, sticky="w")
        ttk.Combobox(crm_box, textvariable=self.crm_iron_type_var, values=IRON_TYPES, state="readonly").grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Label(crm_box, text="Число зерен").grid(row=2, column=0, sticky="w")
        ttk.Entry(crm_box, textvariable=self.crm_grains_var).grid(row=2, column=1, sticky="ew")
        ttk.Label(crm_box, text="Искажение 0..1").grid(row=3, column=0, sticky="w")
        ttk.Entry(crm_box, textvariable=self.crm_distortion_var).grid(row=3, column=1, sticky="ew")
        ttk.Label(crm_box, text="Температура для диаграммы, C").grid(row=4, column=0, sticky="w")
        ttk.Entry(crm_box, textvariable=self.crm_temp_var).grid(row=4, column=1, sticky="ew")

        phase_box = ttk.LabelFrame(parent, text="Параметры конструктора фаз", padding=8)
        phase_box.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        phase_box.columnconfigure(1, weight=1)

        self.system_var = tk.StringVar(value="fe-c")
        self.stage_var = tk.StringVar(value="auto")
        self.temp_var = tk.DoubleVar(value=20.0)
        self.cooling_var = tk.StringVar(value="equilibrium")
        self.deformation_var = tk.DoubleVar(value=0.0)
        self.aging_temp_var = tk.DoubleVar(value=180.0)
        self.aging_hours_var = tk.DoubleVar(value=8.0)

        ttk.Label(phase_box, text="Система").grid(row=0, column=0, sticky="w")
        combo_system = ttk.Combobox(phase_box, textvariable=self.system_var, values=PHASE_SYSTEMS, state="readonly")
        combo_system.grid(row=0, column=1, sticky="ew")
        combo_system.bind("<<ComboboxSelected>>", lambda _e: self._update_stage_values())
        ttk.Label(phase_box, text="Стадия").grid(row=1, column=0, sticky="w")
        self.stage_combo = ttk.Combobox(phase_box, textvariable=self.stage_var, state="readonly")
        self.stage_combo.grid(row=1, column=1, sticky="ew")
        ttk.Label(phase_box, text="Температура, C").grid(row=2, column=0, sticky="w")
        ttk.Entry(phase_box, textvariable=self.temp_var).grid(row=2, column=1, sticky="ew")
        ttk.Label(phase_box, text="Режим охлаждения").grid(row=3, column=0, sticky="w")
        ttk.Combobox(phase_box, textvariable=self.cooling_var, values=COOLING_MODES, state="readonly").grid(
            row=3, column=1, sticky="ew"
        )
        ttk.Label(phase_box, text="Деформация, %").grid(row=4, column=0, sticky="w")
        ttk.Entry(phase_box, textvariable=self.deformation_var).grid(row=4, column=1, sticky="ew")
        ttk.Label(phase_box, text="Температура старения, C").grid(row=5, column=0, sticky="w")
        ttk.Entry(phase_box, textvariable=self.aging_temp_var).grid(row=5, column=1, sticky="ew")
        ttk.Label(phase_box, text="Старение, ч").grid(row=6, column=0, sticky="w")
        ttk.Entry(phase_box, textvariable=self.aging_hours_var).grid(row=6, column=1, sticky="ew")

        comp_box = ttk.LabelFrame(parent, text="Состав (мас.%)", padding=8)
        comp_box.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        for idx in range(4):
            comp_box.columnconfigure(idx * 2 + 1, weight=1)

        self.comp_vars: dict[str, tk.DoubleVar] = {}
        elements = ["Fe", "C", "Si", "Cu", "Zn", "Al", "Mg", "Mn"]
        for idx, element in enumerate(elements):
            r = idx // 4
            c = (idx % 4) * 2
            ttk.Label(comp_box, text=element).grid(row=r, column=c, sticky="w")
            var = tk.DoubleVar(value=0.0)
            self.comp_vars[element] = var
            ttk.Entry(comp_box, textvariable=var, width=9).grid(row=r, column=c + 1, sticky="ew", padx=(2, 10))

        output_box = ttk.LabelFrame(parent, text="Вывод", padding=8)
        output_box.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        output_box.columnconfigure(1, weight=1)

        self.output_dir_var = tk.StringVar(value=str(Path("examples") / "factory_output"))
        self.prefix_var = tk.StringVar(value="sample")
        self.count_var = tk.IntVar(value=10)
        self.seed_var = tk.IntVar(value=6000)
        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        self.magnification_var = tk.IntVar(value=400)
        self.focus_var = tk.DoubleVar(value=0.96)

        ttk.Label(output_box, text="Папка").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_box, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(output_box, text="Обзор", command=self._browse_output_dir).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(output_box, text="Префикс").grid(row=1, column=0, sticky="w")
        ttk.Entry(output_box, textvariable=self.prefix_var).grid(row=1, column=1, sticky="ew")
        ttk.Label(output_box, text="Количество").grid(row=1, column=2, sticky="w", padx=(6, 0))
        ttk.Entry(output_box, textvariable=self.count_var, width=6).grid(row=1, column=3, sticky="w")
        ttk.Label(output_box, text="Начальный seed").grid(row=2, column=0, sticky="w")
        ttk.Entry(output_box, textvariable=self.seed_var).grid(row=2, column=1, sticky="ew")
        ttk.Label(output_box, text="W x H").grid(row=2, column=2, sticky="w", padx=(6, 0))
        wh = ttk.Frame(output_box)
        wh.grid(row=2, column=3, sticky="w")
        ttk.Entry(wh, textvariable=self.width_var, width=6).grid(row=0, column=0)
        ttk.Label(wh, text="x").grid(row=0, column=1, padx=2)
        ttk.Entry(wh, textvariable=self.height_var, width=6).grid(row=0, column=2)
        ttk.Label(output_box, text="Увеличение микроскопа").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            output_box,
            textvariable=self.magnification_var,
            values=[100, 200, 400, 600],
            state="readonly",
            width=10,
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(output_box, text="Фокус").grid(row=3, column=2, sticky="w", padx=(6, 0))
        ttk.Entry(output_box, textvariable=self.focus_var, width=8).grid(row=3, column=3, sticky="w")

        actions = ttk.Frame(parent)
        actions.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)
        ttk.Button(actions, text="Предпросмотр", command=self.preview_single).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Обновить диаграмму", command=self.update_diagram).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(actions, text="Сгенерировать пакет", command=self.generate_batch).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(actions, text="Сохранить конфиг", command=self.save_config).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        self.status_var = tk.StringVar(value="Готово.")
        ttk.Label(parent, textvariable=self.status_var).grid(row=7, column=0, sticky="ew", pady=(8, 0))

        self._update_stage_values()
        self._load_default_composition()

    def _build_preview(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Предпросмотр образца", padding=8)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(frame)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _build_diagram(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Подробная диаграмма состояния", padding=8)
        frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.diagram_label = ttk.Label(frame)
        self.diagram_label.grid(row=0, column=0, sticky="nsew")

    def _load_preset_names(self) -> None:
        names = [p.stem for p in self.engine.list_preset_paths()]
        self.preset_combo["values"] = names
        if names:
            self.preset_var.set(names[0])

    def _update_stage_values(self) -> None:
        stages = ["auto"] + supported_stages(self.system_var.get().strip())
        self.stage_combo["values"] = stages
        if self.stage_var.get() not in stages:
            self.stage_var.set("auto")

    def _load_default_composition(self) -> None:
        defaults = {"Fe": 99.6, "C": 0.4, "Si": 0.0, "Cu": 0.0, "Zn": 0.0, "Al": 0.0, "Mg": 0.0, "Mn": 0.0}
        for key, value in defaults.items():
            self.comp_vars[key].set(value)

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку вывода")
        if folder:
            self.output_dir_var.set(folder)

    def _composition(self) -> dict[str, float]:
        comp: dict[str, float] = {}
        for key, var in self.comp_vars.items():
            value = float(var.get())
            if value > 0:
                comp[key] = value
        return comp

    def _phase_overrides(self) -> dict[str, Any]:
        return {
            "system": self.system_var.get().strip(),
            "stage": self.stage_var.get().strip(),
            "temperature_c": float(self.temp_var.get()),
            "cooling_mode": self.cooling_var.get().strip(),
            "deformation_pct": float(self.deformation_var.get()),
            "aging_temperature_c": float(self.aging_temp_var.get()),
            "aging_hours": float(self.aging_hours_var.get()),
        }

    def _microscope_overrides(self) -> dict[str, Any]:
        return {
            "magnification": int(self.magnification_var.get()),
            "focus": float(self.focus_var.get()),
            "brightness": 1.0,
            "contrast": 1.08,
            "noise_sigma": 2.0,
            "pan_x": 0.5,
            "pan_y": 0.5,
            "vignette_strength": 0.12,
            "uneven_strength": 0.08,
        }

    def _size(self) -> tuple[int, int]:
        return max(128, int(self.height_var.get())), max(128, int(self.width_var.get()))

    def _phase_template(self) -> MaterialPreset:
        for path in self.engine.list_preset_paths():
            preset = self.engine.load_preset(path)
            if preset.generator.lower().strip() in {"phase_map", "phase", "alloy_phase"}:
                return preset
        return MaterialPreset(
            name="phase_factory_template",
            material="Phase template",
            lab="factory",
            generator="phase_map",
            image_size=(2048, 2048),
            seed=1000,
            composition={"Fe": 99.6, "C": 0.4},
            generation={
                "system": "fe-c",
                "stage": "auto",
                "temperature_c": 20.0,
                "cooling_mode": "equilibrium",
                "deformation_pct": 0.0,
                "aging_temperature_c": 180.0,
                "aging_hours": 8.0,
            },
            microscope={},
            metadata={},
        )

    def _build_sample(self, seed: int) -> FactorySample:
        mode = self.mode_var.get().strip()
        size = self._size()

        if mode == "crm_fe_c":
            image, fractions = generate_crm_fe_c_rgb(
                width=size[1],
                height=size[0],
                carbon_pct=float(self.crm_carbon_var.get()),
                grains_count=max(2, int(self.crm_grains_var.get())),
                seed=seed,
                iron_type=self.crm_iron_type_var.get().strip(),
                distortion_level=float(self.crm_distortion_var.get()),
            )
            metadata = {
                "generator_mode": "crm_fe_c",
                "seed": seed,
                "size": [size[0], size[1]],
                "carbon_wt": float(self.crm_carbon_var.get()),
                "iron_type": self.crm_iron_type_var.get().strip(),
                "grains": int(self.crm_grains_var.get()),
                "distortion_level": float(self.crm_distortion_var.get()),
                "fractions": fractions,
                "fraction_summary": format_fraction_summary(fractions),
            }
            return FactorySample(
                image=image,
                metadata=metadata,
                sample_metadata={"system": "fe-c", "resolved_stage": "room_temperature_structure", "phase_fractions": fractions},
                mode="crm_fe_c",
                source_name="crm_fe_c",
            )

        if mode == "preset":
            preset_name = self.preset_var.get().strip()
            if not preset_name:
                raise ValueError("Preset not selected")
            result = self.engine.generate_from_preset(
                preset_name,
                seed_override=seed,
                image_size_override=size,
                microscope_overrides=self._microscope_overrides(),
                output_size=size,
            )
            self.preview_engine_result = result
            image = result.view_image if result.view_image is not None else result.sample_image
            return FactorySample(
                image=image,
                metadata=result.metadata_for_export(),
                sample_metadata=result.sample_metadata,
                mode="preset",
                source_name=preset_name,
            )

        template = self._phase_template()
        result = self.engine.generate_sample(
            preset=template,
            seed_override=seed,
            image_size_override=size,
            composition_override=self._composition(),
            generation_overrides=self._phase_overrides(),
        )
        self.engine.render_view(result, microscope_overrides=self._microscope_overrides(), output_size=size)
        self.preview_engine_result = result
        image = result.view_image if result.view_image is not None else result.sample_image
        return FactorySample(
            image=image,
            metadata=result.metadata_for_export(),
            sample_metadata=result.sample_metadata,
            mode="phase",
            source_name=self.system_var.get().strip(),
        )

    def preview_single(self) -> None:
        try:
            seed = int(self.seed_var.get())
            sample = self._build_sample(seed)
            self.preview_sample = sample
            self._show_preview(sample.image)
            self.update_diagram()
            self.status_var.set("Предпросмотр сгенерирован.")
        except Exception as exc:
            messagebox.showerror("Генератор образцов", str(exc))

    def _show_preview(self, image: np.ndarray) -> None:
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=2)
        else:
            rgb = image
        pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        self.tk_preview = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=self.tk_preview)

    def _diagram_context(self) -> tuple[str, dict[str, float], float, float, dict[str, float] | None]:
        mode = self.mode_var.get().strip()
        if mode == "crm_fe_c":
            composition = {"C": float(self.crm_carbon_var.get()), "Fe": max(0.0, 100.0 - float(self.crm_carbon_var.get()))}
            temp = float(self.crm_temp_var.get())
            fractions = None
            if self.preview_sample is not None:
                fractions = self.preview_sample.metadata.get("fractions")
            return "fe-c", composition, temp, float(self.aging_hours_var.get()), fractions

        if mode == "phase":
            system = self.system_var.get().strip()
            return system, self._composition(), float(self.temp_var.get()), float(self.aging_hours_var.get()), None

        preset_name = self.preset_var.get().strip()
        if not preset_name:
            return "fe-c", {}, 20.0, 8.0, None
        preset = self.engine.load_preset(preset_name)
        composition = dict(preset.composition)
        temp = float(preset.generation.get("temperature_c", 20.0))
        system = infer_system_from_context(preset.material, preset.generator, composition)
        fractions = None
        if self.preview_sample is not None:
            candidate = self.preview_sample.sample_metadata.get("phase_fractions")
            if isinstance(candidate, dict):
                fractions = {k: float(v) for k, v in candidate.items()}
        return system, composition, temp, float(preset.generation.get("aging_hours", 8.0)), fractions

    def update_diagram(self) -> None:
        try:
            system, composition, temp, aging_hours, fractions = self._diagram_context()
            pil = render_detailed_diagram(
                system=system,
                composition=composition,
                temperature_c=temp,
                aging_hours=aging_hours,
                fractions=fractions,
                size=(860, 390),
            )
            self.tk_diagram = ImageTk.PhotoImage(pil)
            self.diagram_label.configure(image=self.tk_diagram)
        except Exception as exc:
            self.status_var.set(f"Ошибка диаграммы: {exc}")

    def generate_batch(self) -> None:
        out_dir = Path(self.output_dir_var.get().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.prefix_var.get().strip() or "sample"
        count = max(1, int(self.count_var.get()))
        seed_start = int(self.seed_var.get())

        rows: list[dict[str, Any]] = []
        generated = 0
        for i in range(count):
            seed = seed_start + i
            try:
                sample = self._build_sample(seed=seed)
                image_name = f"{prefix}_{i + 1:03d}.png"
                image_path = out_dir / image_name
                save_image(sample.image, image_path)

                meta_path = out_dir / f"{prefix}_{i + 1:03d}.json"
                save_json(sample.metadata, meta_path)

                rows.append(
                    {
                        "index": i + 1,
                        "mode": sample.mode,
                        "source": sample.source_name,
                        "seed": seed,
                        "image": str(image_path),
                        "metadata": str(meta_path),
                        "stage": sample.sample_metadata.get("resolved_stage", ""),
                        "system": sample.sample_metadata.get("system", ""),
                        "error": "",
                    }
                )
                generated += 1
            except Exception as exc:
                rows.append(
                    {
                        "index": i + 1,
                        "mode": self.mode_var.get(),
                        "source": self.preset_var.get() if self.mode_var.get() == "preset" else self.system_var.get(),
                        "seed": seed,
                        "image": "",
                        "metadata": "",
                        "stage": "",
                        "system": "",
                        "error": str(exc),
                    }
                )

        index_path = out_dir / f"{prefix}_index.csv"
        save_measurements_csv(rows, index_path)
        self.status_var.set(f"Пакет завершен: {generated}/{count}. Индекс: {index_path}")
        messagebox.showinfo("Генератор образцов", f"Сгенерировано {generated}/{count} образцов.\nИндекс: {index_path}")

    def save_config(self) -> None:
        payload = {
            "mode": self.mode_var.get(),
            "preset": self.preset_var.get(),
            "crm_fe_c": {
                "carbon": float(self.crm_carbon_var.get()),
                "iron_type": self.crm_iron_type_var.get(),
                "grains": int(self.crm_grains_var.get()),
                "distortion": float(self.crm_distortion_var.get()),
                "diagram_temperature_c": float(self.crm_temp_var.get()),
            },
            "phase": self._phase_overrides(),
            "composition_wt": self._composition(),
            "output": {
                "directory": self.output_dir_var.get(),
                "prefix": self.prefix_var.get(),
                "count": int(self.count_var.get()),
                "seed_start": int(self.seed_var.get()),
                "size": [int(self.height_var.get()), int(self.width_var.get())],
                "magnification": int(self.magnification_var.get()),
                "focus": float(self.focus_var.get()),
            },
        }
        path = filedialog.asksaveasfilename(
            title="Сохранить конфигурацию генератора",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        save_json(payload, path)
        self.status_var.set(f"Конфигурация сохранена: {path}")


def launch_sample_factory(presets_dir: str | Path | None = None) -> None:
    root = tk.Tk()
    app = SampleFactoryApp(root=root, presets_dir=presets_dir)
    _ = app
    root.mainloop()
