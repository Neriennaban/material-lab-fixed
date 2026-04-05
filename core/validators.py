from __future__ import annotations

from typing import Any, Callable


class ValidationError(Exception):
    """Validation error exception"""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class ValidationResult:
    """Result of validation"""

    def __init__(self):
        self.errors: list[ValidationError] = []
        self.warnings: list[str] = []

    def add_error(self, field: str, message: str) -> None:
        """Add validation error"""
        self.errors.append(ValidationError(field, message))

    def add_warning(self, message: str) -> None:
        """Add validation warning"""
        self.warnings.append(message)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return len(self.errors) == 0

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed"""
        if not self.is_valid:
            raise ValidationError("validation", f"{len(self.errors)} validation error(s)")


class Validator:
    """Base validator class"""

    def __init__(self, field: str):
        self.field = field

    def validate(self, value: Any, result: ValidationResult) -> None:
        """Validate value"""
        raise NotImplementedError


class RequiredValidator(Validator):
    """Validate that value is not None or empty"""

    def validate(self, value: Any, result: ValidationResult) -> None:
        if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
            result.add_error(self.field, "Field is required")


class RangeValidator(Validator):
    """Validate that value is within range"""

    def __init__(self, field: str, min_value: float | None = None, max_value: float | None = None):
        super().__init__(field)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any, result: ValidationResult) -> None:
        if value is None:
            return
        
        try:
            num_value = float(value)
            if self.min_value is not None and num_value < self.min_value:
                result.add_error(self.field, f"Value must be >= {self.min_value}")
            if self.max_value is not None and num_value > self.max_value:
                result.add_error(self.field, f"Value must be <= {self.max_value}")
        except (TypeError, ValueError):
            result.add_error(self.field, "Value must be a number")


class LengthValidator(Validator):
    """Validate string or collection length"""

    def __init__(self, field: str, min_length: int | None = None, max_length: int | None = None):
        super().__init__(field)
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any, result: ValidationResult) -> None:
        if value is None:
            return
        
        try:
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                result.add_error(self.field, f"Length must be >= {self.min_length}")
            if self.max_length is not None and length > self.max_length:
                result.add_error(self.field, f"Length must be <= {self.max_length}")
        except TypeError:
            result.add_error(self.field, "Value must have length")


class PatternValidator(Validator):
    """Validate value matches pattern"""

    def __init__(self, field: str, pattern: str):
        super().__init__(field)
        import re
        self.pattern = re.compile(pattern)

    def validate(self, value: Any, result: ValidationResult) -> None:
        if value is None:
            return
        
        if not self.pattern.match(str(value)):
            result.add_error(self.field, f"Value does not match pattern")


class ChoiceValidator(Validator):
    """Validate value is in allowed choices"""

    def __init__(self, field: str, choices: list[Any]):
        super().__init__(field)
        self.choices = choices

    def validate(self, value: Any, result: ValidationResult) -> None:
        if value is None:
            return
        
        if value not in self.choices:
            result.add_error(self.field, f"Value must be one of: {', '.join(map(str, self.choices))}")


class CustomValidator(Validator):
    """Custom validation function"""

    def __init__(self, field: str, validate_fn: Callable[[Any], bool], message: str):
        super().__init__(field)
        self.validate_fn = validate_fn
        self.message = message

    def validate(self, value: Any, result: ValidationResult) -> None:
        if value is None:
            return
        
        try:
            if not self.validate_fn(value):
                result.add_error(self.field, self.message)
        except Exception as e:
            result.add_error(self.field, f"Validation error: {e}")


class CompositionValidator:
    """Validate chemical composition"""

    @staticmethod
    def validate(composition: dict[str, float]) -> ValidationResult:
        """Validate composition dictionary"""
        result = ValidationResult()
        
        if not composition:
            result.add_error("composition", "Composition cannot be empty")
            return result
        
        # Check total percentage
        total = sum(composition.values())
        if abs(total - 100.0) > 0.1:
            result.add_warning(f"Composition total is {total:.2f}%, expected 100%")
        
        # Check individual elements
        for element, value in composition.items():
            if not isinstance(element, str) or not element.strip():
                result.add_error("composition", f"Invalid element name: {element}")
            
            if not isinstance(value, (int, float)):
                result.add_error("composition", f"Invalid value for {element}: {value}")
            elif value < 0:
                result.add_error("composition", f"Negative value for {element}: {value}")
            elif value > 100:
                result.add_error("composition", f"Value exceeds 100% for {element}: {value}")
        
        return result


class ThermalProgramValidator:
    """Validate thermal program"""

    @staticmethod
    def validate(program: list[dict[str, float]]) -> ValidationResult:
        """Validate thermal program points"""
        result = ValidationResult()
        
        if not program:
            result.add_error("thermal_program", "Program cannot be empty")
            return result
        
        prev_time = -1
        for i, point in enumerate(program):
            # Check required fields
            if "time_s" not in point:
                result.add_error(f"point_{i}", "Missing time_s field")
            if "temperature_c" not in point:
                result.add_error(f"point_{i}", "Missing temperature_c field")
            
            # Check time ordering
            if "time_s" in point:
                time_s = point["time_s"]
                if time_s <= prev_time:
                    result.add_error(f"point_{i}", f"Time must be increasing (got {time_s}s after {prev_time}s)")
                prev_time = time_s
            
            # Check temperature range
            if "temperature_c" in point:
                temp = point["temperature_c"]
                if temp < -273.15:
                    result.add_error(f"point_{i}", f"Temperature below absolute zero: {temp}°C")
                if temp > 3000:
                    result.add_warning(f"Temperature very high at point {i}: {temp}°C")
        
        return result
