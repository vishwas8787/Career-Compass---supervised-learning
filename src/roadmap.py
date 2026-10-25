
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_profiles():
    """Load processed occupation profiles."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Occupation features not found. "
            "Run preprocessing.py and feature_engineering.py first."
        )

    return pd.read_csv(DATA_FILE)


def find_occupation(profiles, occupation_code=None, title=None):
    """Find an occupation by its O*NET code or title."""
    if occupation_code:
        match = profiles[
            profiles["O*NET-SOC Code"].astype(str) == str(occupation_code)
        ]
    elif title:
        match = profiles[
            profiles["Title"].str.casefold() == title.strip().casefold()
        ]
    else:
        raise ValueError("Provide an occupation code or title.")

    if match.empty:
        raise ValueError("Occupation not found in the dataset.")

    return match.iloc[0]


def get_skill_gaps(occupation, student_skills):
    """Compare self-reported skills with O*NET importance ratings."""
    gaps = []

    for column in occupation.index:
        if not column.startswith("skill_"):
            continue

        skill_name = column.removeprefix("skill_")
        required_importance = occupation[column]

        if pd.isna(required_importance):
            continue

        current_level = student_skills.get(skill_name, 0)

        gaps.append({
            "skill": skill_name,
            "importance": float(required_importance),
            "current_level": float(current_level),
            "gap": max(float(required_importance) - float(current_level), 0),
        })

    return sorted(
        gaps,
        key=lambda item: (item["gap"], item["importance"]),
        reverse=True,
    )


def generate_roadmap(occupation, student_skills):
    """Create a structured, occupation-specific learning roadmap."""
    gaps = get_skill_gaps(occupation, student_skills)

    priority_skills = [
        item["skill"] for item in gaps
        if item["gap"] > 0
    ][:5]

    zone = occupation.get("Job Zone")

    stages = [
        {
            "stage": "1. Education and foundations",
            "actions": [
                "Review the education and preparation typically associated "
                "with this occupation.",
                "Strengthen relevant foundational knowledge before moving "
                "to advanced material.",
            ],
        },
        {
            "stage": "2. Skill development",
            "actions": [
                f"Prioritize developing {skill}."
                for skill in priority_skills
            ] or [
                "Review the occupation's required skills and deepen "
                "your existing knowledge."
            ],
        },
        {
            "stage": "3. Practical experience",
            "actions": [
                "Complete a small project that applies the target skills.",
                "Build a larger project demonstrating independent "
                "problem-solving.",
                "Document your work and explain your design decisions."
            ],
        },
        {
            "stage": "4. Career preparation",
            "actions": [
                "Build a portfolio demonstrating relevant skills.",
                "Review entry-level job descriptions and their requirements.",
                "Prepare a resume and practise relevant interview questions.",
            ],
        },
    ]

    return {
        "occupation": occupation["Title"],
        "occupation_code": occupation["O*NET-SOC Code"],
        "job_zone": int(zone) if pd.notna(zone) else None,
        "skill_gaps": gaps,
        "stages": stages,
    }


def print_roadmap(roadmap):
    """Display the generated roadmap."""
    print("\n" + "=" * 60)
    print(f"CAREER ROADMAP: {roadmap['occupation']}")
    print("=" * 60)

    print(f"Occupation code: {roadmap['occupation_code']}")
    print(f"O*NET Job Zone: {roadmap['job_zone']}")

    print("\nPriority skill gaps:")

    for item in roadmap["skill_gaps"][:5]:
        print(
            f"- {item['skill']}: "
            f"current={item['current_level']:.1f}, "
            f"importance={item['importance']:.2f}, "
            f"gap={item['gap']:.2f}"
        )

    for stage in roadmap["stages"]:
        print(f"\n{stage['stage']}")

        for action in stage["actions"]:
            print(f"  - {action}")


def main():
    print("CAREERCOMPASS - ROADMAP GENERATOR")

    profiles = load_profiles()

    # Example student profile for initial testing.
    student_skills = {
        "Critical Thinking": 3,
        "Mathematics": 2,
        "Reading Comprehension": 4,
        "Writing": 3,
        "Science": 2,
    }

    # Example occupation; we'll connect this to recommendations later.
    occupation = find_occupation(
        profiles,
        title="Geographers",
    )

    roadmap = generate_roadmap(
        occupation,
        student_skills,
    )

    print_roadmap(roadmap)


if __name__ == "__main__":
    main()
