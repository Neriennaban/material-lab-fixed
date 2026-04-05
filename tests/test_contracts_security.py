import unittest
from core.contracts_v3 import StudentDataV3, TeacherAnswersV3

class TestSecurityContracts(unittest.TestCase):
    def test_student_data_creation(self):
        student_data = StudentDataV3(
            sample_id="test123",
            timestamp="2026-03-05T10:00:00",
            composition_wt={"Fe": 99.68, "C": 0.32},
            thermal_program={},
            prep_route={},
            etch_profile={},
            seed=42,
            resolution=(1024, 768),
            image_sha256="abc123def456"
        )
        self.assertEqual(student_data.sample_id, "test123")
        self.assertEqual(student_data.image_sha256, "abc123def456")

    def test_teacher_answers_creation(self):
        answers = TeacherAnswersV3(
            sample_id="test123",
            image_sha256="abc123def456",
            phase_fractions={"ferrite": 0.69, "pearlite": 0.31},
            inferred_system="Fe-C",
            steel_grade="30",
            carbon_content_calculated=0.32
        )
        self.assertEqual(answers.steel_grade, "30")
        self.assertEqual(answers.carbon_content_calculated, 0.32)
