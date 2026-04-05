"""
Тесты для новых возможностей V4
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from core.cache_manager import AdvancedCache
from core.performance import PerformanceMonitor, BatchProcessor
from core.validators import (
    ValidationResult,
    RequiredValidator,
    RangeValidator,
    LengthValidator,
    ChoiceValidator,
    CompositionValidator,
    ThermalProgramValidator,
)


class TestAdvancedCache(unittest.TestCase):
    """Тесты для продвинутого кэша"""

    def setUp(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = AdvancedCache(
            cache_dir=self.temp_dir,
            max_memory_items=10,
            default_ttl=None,
            enable_disk=True,
        )

    def tearDown(self):
        """Очистка после каждого теста"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_get_set(self):
        """Тест базовых операций get/set"""
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

    def test_dict_key(self):
        """Тест использования словаря как ключа"""
        key = {"param1": 10, "param2": "test"}
        self.cache.set(key, "result")
        self.assertEqual(self.cache.get(key), "result")

    def test_get_or_compute(self):
        """Тест get_or_compute"""
        call_count = [0]

        def compute():
            call_count[0] += 1
            return "computed_value"

        # Первый вызов - вычисление
        result1 = self.cache.get_or_compute("key", compute)
        self.assertEqual(result1, "computed_value")
        self.assertEqual(call_count[0], 1)

        # Второй вызов - из кэша
        result2 = self.cache.get_or_compute("key", compute)
        self.assertEqual(result2, "computed_value")
        self.assertEqual(call_count[0], 1)  # Не вызывалось повторно

    def test_ttl(self):
        """Тест TTL (Time To Live)"""
        import time

        self.cache.set("key", "value", ttl=1)  # 1 секунда
        self.assertEqual(self.cache.get("key"), "value")

        time.sleep(1.1)  # Ждем истечения TTL
        self.assertIsNone(self.cache.get("key"))

    def test_stats(self):
        """Тест статистики кэша"""
        self.cache.get("nonexistent")  # miss
        self.cache.set("key", "value")
        self.cache.get("key")  # hit

        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["total_requests"], 2)


class TestPerformanceMonitor(unittest.TestCase):
    """Тесты для мониторинга производительности"""

    def setUp(self):
        """Настройка перед каждым тестом"""
        self.monitor = PerformanceMonitor()

    def test_measure(self):
        """Тест измерения времени"""
        with self.monitor.measure("test_op"):
            sum(range(1000))

        stats = self.monitor.get_stats("test_op")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["mean"], 0)

    def test_multiple_measurements(self):
        """Тест множественных измерений"""
        for _ in range(5):
            with self.monitor.measure("test_op"):
                sum(range(100))

        stats = self.monitor.get_stats("test_op")
        self.assertEqual(stats["count"], 5)
        self.assertGreater(stats["total"], 0)

    def test_counters(self):
        """Тест счетчиков"""
        self.monitor.increment("counter1")
        self.monitor.increment("counter1")
        self.monitor.increment("counter2", value=5)

        all_stats = self.monitor.get_all_stats()
        self.assertEqual(all_stats["counters"]["counter1"], 2)
        self.assertEqual(all_stats["counters"]["counter2"], 5)


class TestBatchProcessor(unittest.TestCase):
    """Тесты для пакетной обработки"""

    def test_batch_processing(self):
        """Тест пакетной обработки"""
        processor = BatchProcessor(batch_size=10)
        items = list(range(25))

        def process_batch(batch):
            return [x * 2 for x in batch]

        results = processor.process(items, process_batch)
        self.assertEqual(len(results), 25)
        self.assertEqual(results[0], 0)
        self.assertEqual(results[24], 48)


class TestValidators(unittest.TestCase):
    """Тесты для валидаторов"""

    def test_required_validator(self):
        """Тест обязательного поля"""
        result = ValidationResult()
        validator = RequiredValidator("field")

        validator.validate(None, result)
        self.assertFalse(result.is_valid)

        result = ValidationResult()
        validator.validate("value", result)
        self.assertTrue(result.is_valid)

    def test_range_validator(self):
        """Тест валидатора диапазона"""
        result = ValidationResult()
        validator = RangeValidator("temp", min_value=0, max_value=100)

        validator.validate(50, result)
        self.assertTrue(result.is_valid)

        result = ValidationResult()
        validator.validate(150, result)
        self.assertFalse(result.is_valid)

    def test_length_validator(self):
        """Тест валидатора длины"""
        result = ValidationResult()
        validator = LengthValidator("text", min_length=5, max_length=10)

        validator.validate("hello", result)
        self.assertTrue(result.is_valid)

        result = ValidationResult()
        validator.validate("hi", result)
        self.assertFalse(result.is_valid)

    def test_choice_validator(self):
        """Тест валидатора выбора"""
        result = ValidationResult()
        validator = ChoiceValidator("status", ["active", "inactive", "pending"])

        validator.validate("active", result)
        self.assertTrue(result.is_valid)

        result = ValidationResult()
        validator.validate("unknown", result)
        self.assertFalse(result.is_valid)

    def test_composition_validator(self):
        """Тест валидатора состава"""
        # Валидный состав
        composition = {"Fe": 98.5, "C": 1.5}
        result = CompositionValidator.validate(composition)
        self.assertTrue(result.is_valid)

        # Невалидный состав (сумма не 100%)
        composition = {"Fe": 90, "C": 5}
        result = CompositionValidator.validate(composition)
        self.assertTrue(result.is_valid)  # Только предупреждение
        self.assertGreater(len(result.warnings), 0)

        # Невалидный состав (отрицательное значение)
        composition = {"Fe": 105, "C": -5}
        result = CompositionValidator.validate(composition)
        self.assertFalse(result.is_valid)

    def test_thermal_program_validator(self):
        """Тест валидатора термопрограммы"""
        # Валидная программа
        program = [
            {"time_s": 0, "temperature_c": 20},
            {"time_s": 100, "temperature_c": 850},
            {"time_s": 200, "temperature_c": 20},
        ]
        result = ThermalProgramValidator.validate(program)
        self.assertTrue(result.is_valid)

        # Невалидная программа (время не возрастает)
        program = [
            {"time_s": 0, "temperature_c": 20},
            {"time_s": 100, "temperature_c": 850},
            {"time_s": 50, "temperature_c": 20},
        ]
        result = ThermalProgramValidator.validate(program)
        self.assertFalse(result.is_valid)

        # Невалидная программа (отсутствует поле)
        program = [
            {"time_s": 0},
            {"temperature_c": 850},
        ]
        result = ThermalProgramValidator.validate(program)
        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()
