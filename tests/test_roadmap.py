"""
Unit tests for the CareerCompass roadmap generation engine.
"""

import unittest
from src.roadmap import (
    load_profiles,
    find_occupation,
    validate_student_skills,
    get_skill_gaps,
    get_knowledge_areas,
    get_course_roadmap,
    get_project_suggestions,
    generate_roadmap,
)


class TestRoadmapEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_profiles()
        cls.occupation = find_occupation(cls.profiles, title="Data Scientists")

    def test_find_occupation_by_title(self):
        occ = find_occupation(self.profiles, title="Data Scientists")
        self.assertEqual(occ["Title"], "Data Scientists")
        self.assertEqual(occ["O*NET-SOC Code"], "15-2051.00")

    def test_find_occupation_not_found(self):
        with self.assertRaises(ValueError):
            find_occupation(self.profiles, title="Intergalactic Captain")

    def test_validate_student_skills(self):
        valid = {"Mathematics": 3.5, "Critical Thinking": "4"}
        cleaned = validate_student_skills(valid)
        self.assertEqual(cleaned["Mathematics"], 3.5)
        self.assertEqual(cleaned["Critical Thinking"], 4.0)

        with self.assertRaises(ValueError):
            validate_student_skills({"Mathematics": 6.5})

    def test_get_skill_gaps(self):
        student_skills = {"Mathematics": 2.0, "Critical Thinking": 4.5}
        gaps = get_skill_gaps(self.occupation, student_skills)
        self.assertIsInstance(gaps, list)
        if gaps:
            self.assertIn("gap", gaps[0])
            self.assertIn("actionable_exercise", gaps[0])
            self.assertGreater(gaps[0]["gap"], 0)

    def test_get_knowledge_areas(self):
        areas = get_knowledge_areas(self.occupation, top_n=5)
        self.assertLessEqual(len(areas), 5)
        self.assertGreaterEqual(len(areas), 1)
        self.assertIn("name", areas[0])
        self.assertIn("importance", areas[0])

    def test_get_course_roadmap(self):
        roadmap = get_course_roadmap("COMPUTER_AND_IT", job_zone=4)

        self.assertIn("stage_1_foundational", roadmap)
        self.assertIn("stage_2_core", roadmap)
        self.assertIn("stage_3_advanced", roadmap)
        self.assertGreater(len(roadmap["stage_1_foundational"]), 0)

    def test_generate_roadmap_structure(self):
        student_skills = {"Mathematics": 2.5, "Critical Thinking": 3.0}
        rm = generate_roadmap(self.occupation, student_skills, predicted_job_zone=4.0)

        self.assertIn("title", rm)
        self.assertIn("code", rm)
        self.assertIn("course_roadmap", rm)
        self.assertIn("gaps", rm)
        self.assertIn("projects", rm)
        self.assertIn("career_milestones", rm)
        self.assertEqual(rm["title"], "Data Scientists")
        self.assertEqual(rm["predicted_job_zone"], 4.0)


if __name__ == "__main__":
    unittest.main()
