import unittest


class AppImportSmokeTests(unittest.TestCase):
    def test_generator_window_imports(self) -> None:
        from ui_qt.sample_factory_window_v3 import launch_sample_factory_app_v3

        self.assertTrue(callable(launch_sample_factory_app_v3))

    def test_microscope_window_imports(self) -> None:
        from ui_qt.microscope_window import launch_microscope_app

        self.assertTrue(callable(launch_microscope_app))


if __name__ == "__main__":
    unittest.main()
