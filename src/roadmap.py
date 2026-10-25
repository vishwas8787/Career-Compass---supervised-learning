
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"


def load_profiles():
    """Load the processed occupation dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Occupation features not found: {DATA_FILE}\n"
            "Run preprocessing and feature engineering first."
        )

    return pd.read_csv(DATA_FILE)


def find_occupation(profiles, occupation_code=None, title=None):
    """Find an occupation using its O*NET code or title."""
    if occupation_code is not None:
        matches = profiles[
            profiles["O*NET-SOC Code"].astype(str).str.strip()
            == str(occupation_code).strip()
        ]
    elif title is not None:
        matches = profiles[
            profiles["Title"].astype(str).str.casefold()
            == str(title).strip().casefold()
        ]
    else:
        raise ValueError("Provide an occupation code or title.")

    if matches.empty:
        raise ValueError(
            f"Occupation not found: {occupation_code or title}"
        )

    return matches.iloc[0]


def validate_student_skills(student_skills):
    """Validate supplied skill ratings without inventing missing ratings."""
    validated = {}

    for name, value in student_skills.items():
        skill_name = str(name).strip()

        try:
            rating = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Rating for '{skill_name}' must be a number from 0 to 5."
            )

        if not 0 <= rating <= 5:
            raise ValueError(
                f"Rating for '{skill_name}' must be between 0 and 5."
            )

        validated[skill_name.casefold()] = (skill_name, rating)

    return validated



def get_skill_gaps(occupation, student_skills):
    """Compare rated student skills with occupation requirements."""
    gaps = []

    for skill_name, student_rating in student_skills.items():
        column = f"skill_{skill_name}"

        if column not in occupation.index:
            continue

        requirement = occupation[column]

        # Ignore missing occupation requirements.
        if pd.isna(requirement):
            continue

        current = float(student_rating)
        required = float(requirement)
        gap = max(required - current, 0.0)

        if gap > 0:
            gaps.append({
                "skill": skill_name,
                "current": current,
                "importance": required,
                "gap": gap,
            })

    gaps.sort(
    key=lambda item: item["gap"] * item["importance"],
    reverse=True,
)

    return gaps


def get_project_suggestions(title):
    """Suggest practical projects relevant to the selected career."""
    career = str(title).casefold()

    if any(word in career for word in (
        "video game", "game designer", "game developer"
    )):
        return [
            "Build a small playable game with a clear objective and win/loss conditions.",
            "Create a prototype level and document its mechanics, pacing, and layout.",
            "Run playtests, record feedback, and improve the design based on results.",
        ]

    if any(word in career for word in (
        "web", "digital interface", "user interface"
    )):
        return [
            "Design and build a responsive website or interactive prototype.",
            "Test the interface with users and improve confusing interactions.",
            "Publish the project with screenshots, design decisions, and source code.",
        ]

    if any(word in career for word in (
        "software", "computer", "database", "systems analyst"
    )):
        return [
            "Build a working application that solves a clearly defined problem.",
            "Add tests, handle errors, and document the system's design.",
            "Publish the project with setup instructions and a short demonstration.",
        ]

    if "designer" in career or "architect" in career:
        return [
            "Create a portfolio project that solves a specific user or client problem.",
            "Produce sketches or prototypes and explain the design decisions.",
            "Gather feedback and present an improved final version.",
        ]

    if "geographer" in career:
        return [
            "Create a map-based project using a geographic dataset.",
            "Analyse a spatial pattern and explain the findings with visualisations.",
            "Document your data sources, methods, limitations, and conclusions.",
        ]

    return [
        "Complete a small project that applies skills relevant to this occupation.",
        "Build a larger project that demonstrates independent problem-solving.",
        "Document your methods, results, and design decisions in a portfolio.",
    ]


def get_job_zone_guidance(job_zone):
    """Give broad preparation guidance based on the O*NET Job Zone."""
    try:
        zone = int(float(job_zone))
    except (TypeError, ValueError):
        zone = None

    guidance = {
        1: "Focus on basic workplace preparation, introductory training, and supervised practice.",
        2: "Build foundational knowledge through short-term training and practical experience.",
        3: "Develop specialised skills through relevant courses, training, and hands-on projects.",
        4: "Consider relevant higher education, advanced coursework, and substantial practical projects.",
        5: "Plan for extensive specialised education or training and advanced practical experience.",
    }

    return guidance.get(
        zone,
        "Review current job descriptions and identify the education and training commonly required."
    )


def generate_roadmap(occupation, student_skills):
    """Generate a career roadmap from occupation data and rated skills."""
    gaps = get_skill_gaps(occupation, student_skills)
    title = occupation["Title"]
    job_zone = occupation.get("Job Zone", "Not available")

    return {
        "title": title,
        "code": occupation.get("O*NET-SOC Code", "Not available"),
        "job_zone": job_zone,
        "education_guidance": get_job_zone_guidance(job_zone),
        "gaps": gaps,
        "projects": get_project_suggestions(title),
    }


def print_roadmap(roadmap):
    """Display the generated roadmap."""
    print("\n" + "=" * 60)
    print(f"CAREER ROADMAP: {roadmap['title']}")
    print("=" * 60)
    print(f"Occupation code: {roadmap['code']}")
    print(f"O*NET Job Zone: {roadmap['job_zone']}")

    print("\n1. Education and foundations")
    print(f"  - {roadmap['education_guidance']}")
    print("  - Compare course options with the requirements in current job postings.")

    print("\n2. Skill development")
    if roadmap["gaps"]:
        print("  Priority skill gaps among the skills you rated:")
        for item in roadmap["gaps"]:
            print(
                f"  - {item['skill']}: current={item['current']:.1f}, "
                f"importance={item['importance']:.2f}, "
                f"gap={item['gap']:.2f}"
            )
    else:
        print("  - No gaps were identified among the skills you rated.")
        print("  - Rate additional relevant skills to get a more complete comparison.")

    print("\n3. Practical experience")
    for project in roadmap["projects"]:
        print(f"  - {project}")

    print("\n4. Career preparation")
    print("  - Build a portfolio showing your most relevant projects.")
    print("  - Review entry-level job descriptions and note recurring requirements.")
    print("  - Prepare a resume that demonstrates your skills with concrete examples.")
    print("  - Practise explaining your projects and problem-solving decisions.")


def main():
    profiles = load_profiles()

    occupation = find_occupation(
        profiles,
        title="Video Game Designers"
    )

    student_skills = {
        "Active Learning": 3,
        "Active Listening": 3,
        "Critical Thinking": 2,
        "Mathematics": 4,
        "Reading Comprehension": 4,
        "Science": 4,
        "Speaking": 0,
        "Writing": 4,
        "Monitoring": 3,
        "Learning Strategies": 2,
    }

    roadmap = generate_roadmap(occupation, student_skills)
    print_roadmap(roadmap)


if __name__ == "__main__":
    main()
