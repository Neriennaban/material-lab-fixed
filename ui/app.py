from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog, messagebox, ttk

from core.pipeline import GenerationEngine, GenerationResult
from export.export_images import save_image
from export.export_tables import save_json


@dataclass(slots=True)
class PresetRow:
    path: Path
    name: str
    material: str
    lab: str
    generator: str


class VirtualMicroscopeApp:
    """
    Redesigned virtual microscope.

    Key UX goals:
    - fast sample choice
    - instrument-style controls (objective turret + stage movement)
    - one-click capture/save workflow
    """

    OBJECTIVE_LEVELS = [100, 200, 400, 600]

    def __init__(self, root: tk.Tk, presets_dir: str | Path | None = None) -> None:
        self.root = root
        self.engine = GenerationEngine(presets_dir=presets_dir)
        self.root.title("Виртуальный металлографический микроскоп")
        self.root.geometry("1560x920")
        self.root.minsize(1260, 760)

        self.current_result: GenerationResult | None = None
        self.current_preset: PresetRow | None = None
        self.current_view: np.ndarray | None = None
        self.tk_view_image: ImageTk.PhotoImage | None = None
        self.tk_nav_image: ImageTk.PhotoImage | None = None
        self._last_view_signature: tuple[Any, ...] | None = None
        self._last_nav_signature: tuple[Any, ...] | None = None

        self.preset_rows: list[PresetRow] = []
        self.filtered_rows: list[PresetRow] = []

        self._configure_style()
        self._build_layout()
        self._load_presets()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        bg = "#18212c"
        panel = "#1f2b39"
        accent = "#f4b942"
        text = "#e9edf2"
        muted = "#aeb8c4"

        self.root.configure(background=bg)
        style.configure(".", background=panel, foreground=text)
        style.configure("TFrame", background=panel)
        style.configure("TLabelframe", background=panel, bordercolor="#2e3c4d")
        style.configure("TLabelframe.Label", background=panel, foreground=muted)
        style.configure("TLabel", background=panel, foreground=text)
        style.configure("TButton", background="#2b3b4e", foreground=text, borderwidth=1)
        style.map("TButton", background=[("active", "#37506a")])
        style.configure("Accent.TButton", background=accent, foreground="#1a1a1a", borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#ffd27a")])
        style.configure("Treeview", background="#243140", fieldbackground="#243140", foreground=text)
        style.map("Treeview", background=[("selected", "#3f5f81")])

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, padding=10)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=0)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        left = ttk.Frame(shell, width=420)
        right = ttk.Frame(shell)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        library = ttk.LabelFrame(parent, text="Библиотека образцов", padding=8)
        library.grid(row=0, column=0, sticky="nsew")
        library.columnconfigure(0, weight=1)
        library.rowconfigure(2, weight=1)

        self.search_var = tk.StringVar()
        search = ttk.Entry(library, textvariable=self.search_var)
        search.grid(row=0, column=0, sticky="ew")
        self.search_var.trace_add("write", lambda *_: self._filter_library())

        columns = ("material", "lab", "gen")
        self.tree = ttk.Treeview(library, columns=columns, show="headings", height=16)
        self.tree.grid(row=2, column=0, sticky="nsew", pady=8)
        self.tree.heading("material", text="Материал")
        self.tree.heading("lab", text="ЛР")
        self.tree.heading("gen", text="Генератор")
        self.tree.column("material", width=190, anchor="w")
        self.tree.column("lab", width=60, anchor="center")
        self.tree.column("gen", width=110, anchor="center")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_library_select())

        buttons = ttk.Frame(library)
        buttons.grid(row=3, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Обновить", command=self._load_presets).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Сгенерировать", style="Accent.TButton", command=self.generate_sample).grid(
            row=0, column=1, sticky="ew"
        )

        info = ttk.LabelFrame(parent, text="Информация об образце", padding=8)
        info.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        info.columnconfigure(0, weight=1)
        self.info_text = tk.Text(info, height=8, wrap="word", bg="#243140", fg="#e9edf2", relief=tk.FLAT)
        self.info_text.grid(row=0, column=0, sticky="ew")
        self.info_text.configure(state=tk.DISABLED)

        controls = ttk.LabelFrame(parent, text="Параметры съемки", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        self.seed_var = tk.IntVar(value=42)
        self.focus_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.noise_var = tk.DoubleVar(value=3.0)
        self.vignette_var = tk.DoubleVar(value=0.15)
        self.uneven_var = tk.DoubleVar(value=0.08)

        row = 0
        ttk.Label(controls, text="Seed").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.seed_var, width=12).grid(row=row, column=1, sticky="ew")
        row += 1
        self._add_scale(controls, row, "Focus", self.focus_var, 0.0, 1.0)
        row += 1
        self._add_scale(controls, row, "Brightness", self.brightness_var, 0.6, 1.6)
        row += 1
        self._add_scale(controls, row, "Contrast", self.contrast_var, 0.6, 1.7)
        row += 1
        self._add_scale(controls, row, "Noise", self.noise_var, 0.0, 12.0)
        row += 1
        self._add_scale(controls, row, "Vignette", self.vignette_var, 0.0, 0.6)
        row += 1
        self._add_scale(controls, row, "Uneven Light", self.uneven_var, 0.0, 0.4)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        header = ttk.LabelFrame(parent, text="Голова микроскопа", padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)
        header.columnconfigure(2, weight=0)

        self.status_var = tk.StringVar(value="Выберите образец и нажмите «Сгенерировать».")
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        objective_frame = ttk.Frame(header)
        objective_frame.grid(row=0, column=1, sticky="e", padx=(8, 8))
        self.mag_var = tk.IntVar(value=200)
        self.obj_buttons: dict[int, ttk.Button] = {}
        for idx, level in enumerate(self.OBJECTIVE_LEVELS):
            btn = ttk.Button(
                objective_frame,
                text=f"{level}x",
                command=lambda lv=level: self._set_objective(lv),
                width=6,
            )
            btn.grid(row=0, column=idx, padx=2)
            self.obj_buttons[level] = btn

        save_bar = ttk.Frame(header)
        save_bar.grid(row=0, column=2, sticky="e")
        ttk.Button(save_bar, text="Обновить", command=self.update_view).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(save_bar, text="Сохранить снимок", style="Accent.TButton", command=self.save_current_image).grid(
            row=0, column=1, padx=(0, 4)
        )
        ttk.Button(save_bar, text="Сохранить метаданные", command=self.save_metadata).grid(row=0, column=2)

        body = ttk.Frame(parent)
        body.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        view_box = ttk.LabelFrame(body, text="Поле зрения", padding=6)
        view_box.grid(row=0, column=0, sticky="nsew")
        view_box.columnconfigure(0, weight=1)
        view_box.rowconfigure(0, weight=1)
        self.view_label = ttk.Label(view_box)
        self.view_label.grid(row=0, column=0, sticky="nsew")
        self.view_label.bind("<MouseWheel>", self._on_mouse_wheel)
        self.view_label.bind("<Configure>", lambda _e: self._rerender_on_resize())

        stage_box = ttk.LabelFrame(body, text="Столик микроскопа", padding=8)
        stage_box.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        stage_box.columnconfigure(0, weight=1)

        self.pan_x_var = tk.DoubleVar(value=0.5)
        self.pan_y_var = tk.DoubleVar(value=0.5)

        nav_frame = ttk.Frame(stage_box)
        nav_frame.grid(row=0, column=0, pady=(0, 8))
        self.nav_canvas = tk.Canvas(
            nav_frame,
            width=240,
            height=240,
            highlightthickness=1,
            highlightbackground="#6f8096",
            bg="#111821",
        )
        self.nav_canvas.grid(row=0, column=0)
        self.nav_canvas.bind("<Button-1>", self._on_navigator_click)

        dpad = ttk.Frame(stage_box)
        dpad.grid(row=1, column=0)
        ttk.Button(dpad, text="Вверх", width=8, command=lambda: self._nudge_stage(0.0, -0.05)).grid(
            row=0, column=1, pady=(0, 4)
        )
        ttk.Button(dpad, text="Влево", width=8, command=lambda: self._nudge_stage(-0.05, 0.0)).grid(row=1, column=0, padx=(0, 4))
        ttk.Button(dpad, text="Центр", width=8, command=self._center_stage).grid(row=1, column=1)
        ttk.Button(dpad, text="Вправо", width=8, command=lambda: self._nudge_stage(0.05, 0.0)).grid(row=1, column=2, padx=(4, 0))
        ttk.Button(dpad, text="Вниз", width=8, command=lambda: self._nudge_stage(0.0, 0.05)).grid(
            row=2, column=1, pady=(4, 0)
        )

    def _add_scale(self, parent: ttk.Frame, row: int, label: str, var: tk.DoubleVar, min_value: float, max_value: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(parent, variable=var, from_=min_value, to=max_value)
        scale.grid(row=row, column=1, sticky="ew")
        scale.bind("<ButtonRelease-1>", lambda _e: self.update_view())

    def _load_presets(self) -> None:
        self.preset_rows.clear()
        for path in self.engine.list_preset_paths():
            try:
                preset = self.engine.load_preset(path)
            except Exception:
                continue
            self.preset_rows.append(
                PresetRow(
                    path=path,
                    name=preset.name,
                    material=preset.material,
                    lab=preset.lab,
                    generator=preset.generator,
                )
            )
        self._filter_library()
        if self.filtered_rows:
            first = self.tree.get_children()
            if first:
                self.tree.selection_set(first[0])
                self._on_library_select()

    def _filter_library(self) -> None:
        needle = self.search_var.get().strip().lower()
        if needle:
            rows = [
                row
                for row in self.preset_rows
                if needle in row.name.lower() or needle in row.material.lower() or needle in row.lab.lower()
            ]
        else:
            rows = list(self.preset_rows)
        self.filtered_rows = rows

        self.tree.delete(*self.tree.get_children())
        for idx, row in enumerate(rows):
            self.tree.insert("", tk.END, iid=str(idx), values=(row.material, row.lab, row.generator))

    def _on_library_select(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        if idx < 0 or idx >= len(self.filtered_rows):
            return
        row = self.filtered_rows[idx]
        self.current_preset = row

        preset = self.engine.load_preset(row.path)
        self.seed_var.set(preset.seed)
        self.mag_var.set(int(preset.microscope.get("magnification", 200)))
        self.focus_var.set(float(preset.microscope.get("focus", 1.0)))
        self.brightness_var.set(float(preset.microscope.get("brightness", 1.0)))
        self.contrast_var.set(float(preset.microscope.get("contrast", 1.0)))
        self.noise_var.set(float(preset.microscope.get("noise_sigma", 3.0)))
        self.vignette_var.set(float(preset.microscope.get("vignette_strength", 0.15)))
        self.uneven_var.set(float(preset.microscope.get("uneven_strength", 0.08)))
        self.pan_x_var.set(float(preset.microscope.get("pan_x", 0.5)))
        self.pan_y_var.set(float(preset.microscope.get("pan_y", 0.5)))
        self._refresh_objective_buttons()

        lines = [
            f"Имя: {preset.name}",
            f"Материал: {preset.material}",
            f"Лабораторная: {preset.lab}",
            f"Генератор: {preset.generator}",
            f"Размер образца: {preset.image_size[1]}x{preset.image_size[0]} px",
        ]
        if preset.composition:
            lines.append("Состав (мас.%):")
            lines.extend([f"  {el} = {value:.3g}" for el, value in preset.composition.items()])
        self._set_info("\n".join(lines))
        self.status_var.set(f"Загружен пресет: {preset.name}.")

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.configure(state=tk.DISABLED)

    def _target_size(self) -> tuple[int, int]:
        h = max(420, self.view_label.winfo_height())
        w = max(420, self.view_label.winfo_width())
        return (h, w)

    def _microscope_params(self) -> dict[str, Any]:
        return {
            "magnification": int(self.mag_var.get()),
            "focus": float(self.focus_var.get()),
            "brightness": float(self.brightness_var.get()),
            "contrast": float(self.contrast_var.get()),
            "noise_sigma": float(self.noise_var.get()),
            "pan_x": float(self.pan_x_var.get()),
            "pan_y": float(self.pan_y_var.get()),
            "vignette_strength": float(self.vignette_var.get()),
            "uneven_strength": float(self.uneven_var.get()),
            "seed": int(self.seed_var.get()) + 1000,
        }

    @staticmethod
    def _q(value: float, precision: int = 4) -> int:
        return int(round(float(value) * (10**precision)))

    def _view_signature(self, output_size: tuple[int, int], params: dict[str, Any]) -> tuple[Any, ...]:
        if self.current_result is None:
            return ()
        sample = self.current_result.sample_image
        return (
            sample.__array_interface__["data"][0],
            int(sample.shape[0]),
            int(sample.shape[1]),
            int(output_size[0]),
            int(output_size[1]),
            int(params.get("magnification", 200)),
            self._q(params.get("focus", 1.0)),
            self._q(params.get("brightness", 1.0), precision=3),
            self._q(params.get("contrast", 1.0), precision=3),
            self._q(params.get("noise_sigma", 3.0)),
            self._q(params.get("pan_x", 0.5)),
            self._q(params.get("pan_y", 0.5)),
            self._q(params.get("vignette_strength", 0.15)),
            self._q(params.get("uneven_strength", 0.08)),
            int(params.get("seed", 0)),
        )

    def generate_sample(self) -> None:
        if self.current_preset is None:
            messagebox.showwarning("Виртуальный микроскоп", "Сначала выберите пресет образца.")
            return
        preset = self.engine.load_preset(self.current_preset.path)
        self.current_result = self.engine.generate_sample(
            preset=preset,
            seed_override=int(self.seed_var.get()),
        )
        self._last_view_signature = None
        self._last_nav_signature = None
        self.update_view()

    def update_view(self) -> None:
        if self.current_result is None:
            return
        output_size = self._target_size()
        params = self._microscope_params()
        signature = self._view_signature(output_size, params)
        if signature == self._last_view_signature and self.current_view is not None:
            return
        self.engine.render_view(
            self.current_result,
            microscope_overrides=params,
            output_size=output_size,
        )
        if self.current_result.view_image is None:
            return
        self._last_view_signature = signature
        self.current_view = self.current_result.view_image
        self._draw_view()
        self._draw_navigator()
        self._update_status_line()

    def _draw_view(self) -> None:
        if self.current_view is None:
            return
        gray = self.current_view
        rgb = np.stack([gray] * 3, axis=2)
        pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

        draw = ImageDraw.Draw(pil)
        w, h = pil.size
        cx = w // 2
        cy = h // 2
        draw.line((cx - 12, cy, cx + 12, cy), fill=(255, 80, 80), width=1)
        draw.line((cx, cy - 12, cx, cy + 12), fill=(255, 80, 80), width=1)

        self.tk_view_image = ImageTk.PhotoImage(pil)
        self.view_label.configure(image=self.tk_view_image)

    def _draw_navigator(self) -> None:
        if self.current_result is None:
            return
        sample = self.current_result.sample_image
        nav_size = 240
        nav_signature = (sample.__array_interface__["data"][0], int(sample.shape[0]), int(sample.shape[1]))
        if nav_signature != self._last_nav_signature:
            pil = Image.fromarray(np.stack([sample] * 3, axis=2).astype(np.uint8), mode="RGB")
            pil = pil.resize((nav_size, nav_size), resample=Image.Resampling.BILINEAR)
            self.tk_nav_image = ImageTk.PhotoImage(pil)
            self._last_nav_signature = nav_signature

        self.nav_canvas.delete("all")
        self.nav_canvas.create_image(0, 0, image=self.tk_nav_image, anchor="nw")

        meta = self.current_result.view_metadata or {}
        origin = meta.get("crop_origin_px")
        crop = meta.get("crop_size_px")
        if isinstance(origin, list) and isinstance(crop, list):
            sh, sw = sample.shape
            oy, ox = float(origin[0]), float(origin[1])
            ch, cw = float(crop[0]), float(crop[1])
            x0 = ox / max(sw, 1) * nav_size
            y0 = oy / max(sh, 1) * nav_size
            x1 = (ox + cw) / max(sw, 1) * nav_size
            y1 = (oy + ch) / max(sh, 1) * nav_size
            self.nav_canvas.create_rectangle(x0, y0, x1, y1, outline="#ff4f4f", width=2)

    def _update_status_line(self) -> None:
        if self.current_result is None:
            return
        meta = self.current_result.view_metadata or {}
        self.status_var.set(
            f"Материал: {self.current_result.preset.material} | "
            f"Объектив: {meta.get('magnification', '-')}x | "
            f"Фокус: {float(meta.get('focus', 1.0)):.2f} | "
            f"Смещение: ({float(self.pan_x_var.get()):.2f}, {float(self.pan_y_var.get()):.2f})"
        )

    def _set_objective(self, level: int) -> None:
        self.mag_var.set(level)
        self._refresh_objective_buttons()
        self.update_view()

    def _refresh_objective_buttons(self) -> None:
        active = int(self.mag_var.get())
        for level, button in self.obj_buttons.items():
            if level == active:
                button.configure(style="Accent.TButton")
            else:
                button.configure(style="TButton")

    def _nudge_stage(self, dx: float, dy: float) -> None:
        self.pan_x_var.set(float(np.clip(self.pan_x_var.get() + dx, 0.0, 1.0)))
        self.pan_y_var.set(float(np.clip(self.pan_y_var.get() + dy, 0.0, 1.0)))
        self.update_view()

    def _center_stage(self) -> None:
        self.pan_x_var.set(0.5)
        self.pan_y_var.set(0.5)
        self.update_view()

    def _on_navigator_click(self, event: tk.Event[Any]) -> None:
        x = float(np.clip(event.x, 0, 239))
        y = float(np.clip(event.y, 0, 239))
        self.pan_x_var.set(x / 239.0)
        self.pan_y_var.set(y / 239.0)
        self.update_view()

    def _on_mouse_wheel(self, event: tk.Event[Any]) -> None:
        levels = self.OBJECTIVE_LEVELS
        current = int(self.mag_var.get())
        idx = levels.index(current) if current in levels else 1
        idx = min(idx + 1, len(levels) - 1) if event.delta > 0 else max(idx - 1, 0)
        self.mag_var.set(levels[idx])
        self._refresh_objective_buttons()
        self.update_view()

    def _rerender_on_resize(self) -> None:
        if self.current_result is not None:
            self.update_view()

    def save_current_image(self) -> None:
        if self.current_view is None:
            messagebox.showwarning("Виртуальный микроскоп", "Нет изображения для сохранения.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить снимок",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tiff")],
        )
        if not path:
            return
        saved = save_image(self.current_view, path)
        self.status_var.set(f"Снимок сохранен: {saved}")

    def save_metadata(self) -> None:
        if self.current_result is None:
            messagebox.showwarning("Виртуальный микроскоп", "Нет метаданных для сохранения.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить метаданные снимка",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        payload = self.current_result.metadata_for_export()
        save_json(payload, path)
        self.status_var.set(f"Метаданные сохранены: {path}")


def launch_app(presets_dir: str | Path | None = None) -> None:
    root = tk.Tk()
    app = VirtualMicroscopeApp(root=root, presets_dir=presets_dir)
    _ = app
    root.mainloop()
