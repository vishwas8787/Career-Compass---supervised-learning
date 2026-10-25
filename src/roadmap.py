
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"


def load_profiles():
    """Load occupation profiles containing O*NET requirements."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Occupation profiles not found: {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


def find_occupation(profiles, occupation_code=None, title=None):
    """Find an occupation by its O*NET code or title."""
    if occupation_code is not None:
        matches = profiles[
            profiles["O*NET-SOC Code"].astype(str)
            == str(occupation_code)
        ]
    elif title is not None:
        matches = profiles[
            profiles["Title"].astype(str).str.casefold()
            == str(title).casefold()
        ]
    else:
        raise ValueError("Provide an occupation code or title.")

    if matches.empty:
        raise ValueError("Occupation not found.")

    return matches.iloc[0]


def validate_student_skills(student_skills):
    """Validate skill ratings supplied by the student."""
    validated = {}

    for name, rating in (student_skills or {}).items():
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            raise ValueError(
                f"Skill rating for '{name}' must be numeric."
            )

        if not 0 <= rating <= 5:
            raise ValueError(
                f"Skill rating for '{name}' must be between 0 and 5."
            )

        validated[name.strip()] = rating

    return validated


def get_skill_gaps(occupation, student_skills):
    """Rank skill gaps using the occupation's actual O*NET requirements."""
    gaps = []

    # Match skill names without depending on capitalization.
    occupation_columns = {
        str(column).casefold(): column
        for column in occupation.index
    }

    for skill_name, student_rating in student_skills.items():
        key = f"skill_{skill_name}".casefold()
        column = occupation_columns.get(key)

        if column is None:
            continue

        requirement = pd.to_numeric(
            occupation[column], errors="coerce"
        )

        if pd.isna(requirement):
            continue

        requirement = float(requirement)
        gap = max(requirement - student_rating, 0.0)

        if gap > 0:
            gaps.append({
                "skill": skill_name,
                "current": student_rating,
                "importance": requirement,
                "gap": gap,
                "priority": gap * requirement,
            })

    gaps.sort(
        key=lambda item: item["priority"],
        reverse=True,
    )

    return gaps


def get_knowledge_areas(occupation, top_n=5):
    """Identify the occupation's highest-rated O*NET knowledge areas."""
    areas = []

    for column in occupation.index:
        if not str(column).startswith("knowledge_"):
            continue

        value = pd.to_numeric(
            occupation[column], errors="coerce"
        )

        if pd.isna(value):
            continue

        areas.append({
            "name": str(column)[len("knowledge_"):],
            "importance": float(value),
        })

    areas.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return areas[:top_n]


def get_job_zone_guidance(job_zone):
    """Explain the typical preparation associated with a Job Zone."""
    guidance = {
        1: [
            "Focus on basic workplace skills and introductory training.",
            "Explore entry-level opportunities and practical experience.",
        ],
        2: [
            "Consider a high-school qualification or equivalent preparation.",
            "Build skills through practical training and supervised experience.",
        ],
        3: [
            "Consider vocational training, relevant coursework, or an apprenticeship.",
            "Practise core occupational skills through structured projects.",
        ],
        4: [
            "Consider relevant higher education or substantial specialised training.",
            "Build a portfolio of projects demonstrating applied competence.",
        ],
        5: [
            "Investigate advanced education and extensive specialised preparation.",
            "Develop advanced practical or research experience in the field.",
        ],
    }

    try:
        zone = int(float(job_zone))
    except (TypeError, ValueError):
        zone = 0

    return guidance.get(
        zone,
        ["Research the education and training commonly required for this occupation."],
    )


def get_project_suggestions(title):
    """Return practical projects tailored to the occupation."""
    title = str(title).casefold()

    if any(word in title for word in (
        "dancer", "choreographer"
    )):
        return [
            "Create and record a short choreography sequence with a clear theme.",
            "Practise a chosen dance technique and document improvement over several weeks.",
            "Perform or rehearse a routine, review the recording, and refine timing and movement.",
        ]

    if any(word in title for word in (
        "fine artist", "illustrator", "painter", "sculptor"
    )):
        return [
            "Create a themed collection of three original artworks.",
            "Experiment with different techniques and document the results.",
            "Assemble a portfolio explaining the concept and process behind each work.",
        ]

    if any(word in title for word in (
        "model", "fashion model"
    )):
        return [
            "Build a professional portfolio with varied poses, expressions, and styling.",
            "Practise movement, posture, and presenting garments on camera.",
            "Review test-shoot feedback and create an improved portfolio selection.",
        ]

    if any(word in title for word in (
        "makeup artist", "make-up artist"
    )):
        return [
            "Create looks for different occasions and document each with photographs.",
            "Practise colour matching, blending, and hygiene procedures.",
            "Build a portfolio showing different styles and the techniques used.",
        ]

    if any(word in title for word in (
        "tailor", "dressmaker", "sewer"
    )):
        return [
            "Produce a simple garment from measurements and a pattern.",
            "Practise alterations and document the fitting and finishing process.",
            "Create a small portfolio showing construction quality and finished garments.",
        ]

    if any(word in title for word in (
        "data scientist", "statistician", "biostatistician",
        "quantitative analyst", "econometrician"
    )):
        return [
            "Choose a real dataset, clean it, and conduct exploratory data analysis.",
            "Build and evaluate a statistical or machine-learning model using appropriate metrics.",
            "Present findings with visualisations, limitations, and reproducible code.",
        ]

    if any(word in title for word in (
        "software developer", "software engineer",
        "web developer", "computer programmer"
    )):
        return [
            "Build a working application that solves a specific problem.",
            "Add input validation, automated checks, and clear documentation.",
            "Publish the project with setup instructions and examples of its use.",
        ]

    if any(word in title for word in (
        "research scientist", "microbiologist",
        "epidemiologist", "researcher"
    )):
        return [
            "Choose a focused research question and review relevant published studies.",
            "Analyse an appropriate dataset or design a safe, feasible investigation.",
            "Write a report covering methods, results, limitations, and conclusions.",
        ]

    if any(word in title for word in (
        "teacher", "instructor", "educator"
    )):
        return [
            "Prepare a lesson plan with measurable learning objectives.",
            "Create a learning activity and a short assessment for those objectives.",
            "Use feedback or assessment results to revise the lesson.",
        ]

    if any(word in title for word in (
        "nurse", "physician", "doctor", "therapist",
        "medical", "healthcare"
    )):
        return [
            "Study a relevant clinical or professional case using reputable educational resources.",
            "Practise applicable knowledge through supervised training or approved simulations.",
            "Prepare a case-based summary that explains decisions and relevant safety considerations.",
        ]

    if any(word in title for word in (
        "engineer", "engineering", "architect"
    )):
        return [
            "Design a small solution to a realistic technical problem.",
            "Create a prototype or simulation and evaluate it against stated requirements.",
            "Document design choices, test results, limitations, and improvements.",
        ]

    if any(word in title for word in (
        "financial", "accountant", "economist", "auditor"
    )):
        return [
            "Analyse a sample financial or economic dataset.",
            "Prepare a report explaining trends, assumptions, and calculations.",
            "Compare alternative scenarios and explain their potential implications.",
        ]

    if any(word in title for word in (
        "designer", "graphic", "multimedia", "animator"
    )):
        return [
            "Create a focused design project with a defined audience and objective.",
            "Produce several concepts and explain the reasoning behind the final choice.",
            "Collect feedback, revise the design, and document the final portfolio piece.",
        ]

    if any(word in title for word in (
        "counselor", "counsellor", "social worker"
    )):
        return [
            "Study a relevant professional case using ethical practice guidelines.",
            "Practise communication and assessment scenarios through approved training.",
            "Reflect on a simulated case and identify appropriate support or referral options.",
        ]

    return [
        f"Research the main responsibilities and required skills of {title}.",
        "Complete a small practical project demonstrating one important occupational skill.",
        "Review the result against relevant professional expectations and document improvements.",
    ]


def generate_roadmap(occupation, student_skills):
    """Build a personalised roadmap from occupation requirements and ratings."""
    student_skills = validate_student_skills(student_skills)

    title = str(occupation["Title"])
    job_zone = occupation.get("Job Zone")

    return {
        "title": title,
        "code": occupation["O*NET-SOC Code"],
        "job_zone": job_zone,
        "education_guidance": get_job_zone_guidance(job_zone),
        "gaps": get_skill_gaps(occupation, student_skills),
        "knowledge_areas": get_knowledge_areas(occupation),
        "projects": get_project_suggestions(title),
    }


def print_roadmap(roadmap):
    """Display the career-specific roadmap."""
    print("\n" + "=" * 60)
    print(f"CAREER ROADMAP: {roadmap['title']}")
    print("=" * 60)
    print(f"Occupation code: {roadmap['code']}")
    print(f"O*NET Job Zone: {roadmap['job_zone']}")

    print("\n1. Education and foundations")
    for item in roadmap["education_guidance"]:
        print(f"  - {item}")

    print("\n2. Skill development")
    gaps = roadmap["gaps"]

    if gaps:
        print("  Priority gaps based on your ratings and occupation requirements:")
        for item in gaps[:5]:
            print(
                f"  - {item['skill']}: current={item['current']:.1f}, "
                f"importance={item['importance']:.2f}, "
                f"gap={item['gap']:.2f}"
            )
    elif roadmap.get("gaps") == []:
        print("  No gaps found among the skills you rated and the available requirements.")

    print("\n3. Knowledge to develop")
    areas = roadmap["knowledge_areas"]

    if areas:
        for area in areas:
            print(
                f"  - {area['name']} "
                f"(O*NET rating: {area['importance']:.2f})"
            )
        print("  These are occupation-level priorities, not an assessment of your current knowledge.")
    else:
        print("  No knowledge ratings are available for this occupation.")

    print("\n4. Practical projects")
    for project in roadmap["projects"]:
        print(f"  - {project}")

    print("\n5. Career preparation")
    print("  - Compare these priorities with current job or training descriptions.")
    print("  - Record evidence of your progress and update your portfolio.")
    print("  - Reassess your skills as you complete relevant practice.")


def main():
    profiles = load_profiles()
    occupation = find_occupation(
        profiles,
        title="Video Game Designers",
    )

    student_skills = {
        "Speaking": 0,
        "Critical Thinking": 2,
        "Active Listening": 3,
        "Learning Strategies": 2,
        "Active Learning": 3,
        "Monitoring": 3,
    }

    roadmap = generate_roadmap(occupation, student_skills)
    print_roadmap(roadmap)


if __name__ == "__main__":
    main()
