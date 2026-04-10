# PyInstaller spec file for the Fe-C virtual microscope viewer.
# Entry point: ``run_app_v2.py`` → ``ui_qt.microscope_window``.
# The microscope does not need presets_v3 / profiles_v3 (it reads
# pre-rendered sample PNGs), but it still lazy-imports a handful of
# modules from ``core.metallography_v3`` that expect
# ``core/rulebook/`` on disk, so the rulebook is shipped alongside.

from pathlib import Path

block_cipher = None
PROJECT_ROOT = Path(".").resolve()

datas = [
    (str(PROJECT_ROOT / "core" / "rulebook"), "core/rulebook"),
]

# Same heavy-dependency blocklist as the generator spec. The viewer
# is a pure Qt app so everything scientific beyond numpy/scipy/PIL
# can be stripped.
excludes = [
    "matplotlib",
    "matplotlib.backends",
    "pandas",
    "jupyter",
    "IPython",
    "notebook",
    "tkinter",
    "test",
    "tests",
    "pytest",
    "unittest",
    "sphinx",
    "docutils",
    "tornado",
    "zmq",
    "setuptools",
    "pip",
    "wheel",
    "lib2to3",
    "pydoc_data",
    "sympy",
    "pycalphad",
    "sklearn",
    "torch",
    "tensorflow",
    "cupy",
    "lpips",
    "torchmetrics",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DAnimation",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtLocation",
    "PySide6.QtPositioning",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtNetwork",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtSql",
    "PySide6.QtUiTools",
    "PySide6.QtHelp",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSvgWidgets",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebSockets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtConcurrent",
    "PySide6.QtPrintSupport",
    "PySide6.QtXml",
    "PySide6.QtOpcUa",
    "PySide6.QtHttpServer",
]

a = Analysis(
    ["run_app_v2.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "ui_qt.microscope_window",
        "ui_qt.spinbox_wheel_filter",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ferro_micro_microscope",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        "vcruntime140.dll",
        "python3*.dll",
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
    ],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
