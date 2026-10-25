"""
CareerCompass - Career & Education Advisor
Command-line Interface & Agent Demo
Course: CSE395 Supervised Learning
"""

import argparse
from src.recommend import recommend_careers
from src.roadmap import (
    load_profiles,
    find_occupation,
    generate_roadmap,
    print_roadmap,
)


INTERESTS = {
    "1": ("practical", "Practical & Hands-on (tools, physical objects, outdoors)"),
    "2": ("analytical", "Analytical & Scientific (solving problems, research, analysis)"),
    "3": ("creative", "Creative & Artistic (design, art, visual media, writing)"),
    "4": ("social", "Social & Helping (teaching, community, advising, healthcare)"),
    "5": ("business", "Business & Leadership (managing, sales, leading projects)"),
    "6": ("organized", "Structured & Detail-Oriented (organization, data, administration)"),
}

SKILL_NAMES = [
    "Active Learning",
    "Active Listening",
    "Critical Thinking",
    "Mathematics",
    "Reading Comprehension",
    "Science",
    "Speaking",
    "Writing",
    "Monitoring",
    "Learning Strategies",
]

PRESET_PROFILES = {
    "1": {
        "name": "Profile 1: Computer Science, Data Systems & AI",
        "interests": {
            "practical": 1.5,
            "analytical": 5.0,
            "creative": 1.0,
            "social": 0.5,
            "business": 2.0,
            "organized": 4.0,
        },
        "skills": {
            "Mathematics": 2.5,
            "Critical Thinking": 3.0,
            "Active Learning": 3.0,
            "Science": 2.0,
            "Speaking": 1.5,
            "Writing": 2.0,
        },
    },
    "2": {
        "name": "Profile 2: Visual Arts & Creative Design",
        "interests": {
            "practical": 3.0,
            "analytical": 2.0,
            "creative": 5.0,
            "social": 1.5,
            "business": 2.0,
            "organized": 2.0,
        },
        "skills": {
            "Writing": 3.0,
            "Critical Thinking": 2.5,
            "Active Listening": 3.0,
            "Speaking": 2.0,
            "Mathematics": 1.5,
        },
    },
    "3": {
        "name": "Profile 3: Healthcare, Medicine & Biomedical Research",
        "interests": {
            "practical": 1.5,
            "analytical": 5.0,
            "creative": 0.5,
            "social": 4.5,
            "business": 1.5,
            "organized": 2.5,
        },
        "skills": {
            "Science": 3.5,
            "Critical Thinking": 3.0,
            "Reading Comprehension": 3.5,
            "Mathematics": 2.5,
            "Active Listening": 3.5,
        },
    },
}


def get_interests_interactive():
    print("\nRate your interests from 0 to 5.")
    print("0 = Not interested, 5 = Very interested")

    for number, (_, label) in INTERESTS.items():
        print(f"  {number}. {label}")

    selected = {}
    for number, (key, label) in INTERESTS.items():
        short_title = label.split("(")[0].strip()
        while True:
            try:
                value = float(input(f"  {short_title} (0-5): "))
                if 0 <= value <= 5:
                    selected[key] = value
                    break
                print("  Enter a number between 0 and 5.")
            except ValueError:
                print("  Please enter a valid number.")

    return selected


def get_skills_interactive():
    print("\nRate your current skills from 0 to 5.")
    print("0 = No experience, 5 = Very proficient (Press Enter to skip)")

    skills = {}
    for name in SKILL_NAMES:
        while True:
            raw_value = input(f"  {name} (0-5, Enter to skip): ").strip()
            if raw_value == "":
                break
            try:
                value = float(raw_value)
                if 0 <= value <= 5:
                    skills[name] = value
                    break
                print("  Enter a number between 0 and 5, or press Enter to skip.")
            except ValueError:
                print("  Please enter a valid number, or press Enter to skip.")

    return skills


def run_pipeline(interests, skills, auto_choice=1):
    print("\n[CareerCompass] Matching student profile using Nearest Neighbors & Random Forest Classifier...")
    recommendations = recommend_careers(
        interests=interests,
        skills=skills,
        top_n=5,
    )

    print("\n" + "=" * 70)
    print("TOP CAREER RECOMMENDATIONS (Ranked by Match Score)")
    print("=" * 70)
    for index, row in recommendations.iterrows():
        print(
            f"  {index + 1}. {row['Title']:<48} | "
            f"Match: {row['Match Score (%)']:.1f}%"
        )

    if auto_choice is not None and 1 <= auto_choice <= len(recommendations):
        choice = auto_choice
        print(f"\n[CareerCompass] Selected Career #{choice}: {recommendations.iloc[choice - 1]['Title']}")
    else:
        while True:
            try:
                choice = int(input(f"\nChoose a career to generate full roadmap (1-{len(recommendations)}): "))
                if 1 <= choice <= len(recommendations):
                    break
                print("Choose a number from the displayed list.")
            except ValueError:
                print("Please enter a valid number.")

    selected = recommendations.iloc[choice - 1]
    profiles = load_profiles()
    occupation = find_occupation(
        profiles,
        occupation_code=selected["O*NET-SOC Code"],
    )

    roadmap = generate_roadmap(
        occupation,
        skills,
        predicted_job_zone=selected.get("Predicted Job Zone"),
    )

    print_roadmap(roadmap)
    return roadmap


def main():
    parser = argparse.ArgumentParser(description="CareerCompass AI Advisor CLI & Demo")
    parser.add_argument("--demo", action="store_true", help="Run automated agent demo using Profile 1")
    parser.add_argument("--persona", choices=["1", "2", "3"], help="Run preset profile (1, 2, or 3)")
    args = parser.parse_args()

    print("=" * 70)
    print("              CAREERCOMPASS — AI CAREER & EDUCATION ADVISOR")
    print("       Personalized Education-to-Career Planning (Courses -> Skills -> Career)")
    print("=" * 70)

    try:
        if args.demo:
            preset = PRESET_PROFILES["1"]
            print(f"\n[DEMO MODE] Running automated demo with: {preset['name']}")
            run_pipeline(preset["interests"], preset["skills"], auto_choice=1)
            print("\nCareerCompass automated demo complete.")
            return

        if args.persona:
            preset = PRESET_PROFILES[args.persona]
            print(f"\nRunning with: {preset['name']}")
            run_pipeline(preset["interests"], preset["skills"], auto_choice=1)
            print("\nCareerCompass session complete.")
            return

        print("\nSelect an option:")
        print("  1. Interactive Questionnaire (Input custom interests and skills)")
        print("  2. Demo Profile 1: Computer Science, Data Systems & AI")
        print("  3. Demo Profile 2: Visual Arts & Creative Design")
        print("  4. Demo Profile 3: Healthcare, Medicine & Biomedical Research")

        mode = input("\nEnter choice (1-4, default 1): ").strip()

        if mode == "2":
            preset = PRESET_PROFILES["1"]
            run_pipeline(preset["interests"], preset["skills"], auto_choice=1)
        elif mode == "3":
            preset = PRESET_PROFILES["2"]
            run_pipeline(preset["interests"], preset["skills"], auto_choice=1)
        elif mode == "4":
            preset = PRESET_PROFILES["3"]
            run_pipeline(preset["interests"], preset["skills"], auto_choice=1)
        else:
            interests = get_interests_interactive()
            skills = get_skills_interactive()
            run_pipeline(interests, skills, auto_choice=None)

        print("\nCareerCompass session complete.")

    except (FileNotFoundError, ValueError, KeyboardInterrupt) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
