"""
CareerCompass - Career & Education Advisor
Command-line Interface & Agent Demo
"""

import sys
import argparse
from src.recommend import recommend_careers
from src.roadmap import (
    load_profiles,
    find_occupation,
    generate_roadmap,
    print_roadmap,
)


INTERESTS = {
    "1": "realistic",
    "2": "investigative",
    "3": "artistic",
    "4": "social",
    "5": "enterprising",
    "6": "conventional",
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
        "name": "Persona 1: Computer Science & AI Aspirant",
        "interests": {
            "realistic": 3.0,
            "investigative": 5.0,
            "artistic": 1.0,
            "social": 1.5,
            "enterprising": 2.0,
            "conventional": 2.5,
        },
        "skills": {
            "Mathematics": 2.5,
            "Critical Thinking": 3.0,
            "Active Learning": 3.0,
            "Science": 2.5,
            "Speaking": 1.5,
            "Writing": 2.0,
        },
    },
    "2": {
        "name": "Persona 2: Digital Media & Creative Game Designer",
        "interests": {
            "realistic": 2.0,
            "investigative": 2.5,
            "artistic": 5.0,
            "social": 2.0,
            "enterprising": 4.0,
            "conventional": 1.5,
        },
        "skills": {
            "Critical Thinking": 2.5,
            "Active Listening": 3.5,
            "Writing": 3.0,
            "Speaking": 3.0,
            "Learning Strategies": 3.0,
            "Mathematics": 1.5,
        },
    },
    "3": {
        "name": "Persona 3: Healthcare & Biotechnology Researcher",
        "interests": {
            "realistic": 2.0,
            "investigative": 5.0,
            "artistic": 1.0,
            "social": 4.5,
            "enterprising": 1.5,
            "conventional": 3.0,
        },
        "skills": {
            "Science": 3.5,
            "Critical Thinking": 3.0,
            "Reading Comprehension": 3.5,
            "Mathematics": 2.0,
            "Active Listening": 3.5,
            "Active Learning": 3.0,
        },
    },
}


def get_interests_interactive():
    print("\nRate your interests from 0 to 5.")
    print("0 = Not interested, 5 = Very interested")

    for number, name in INTERESTS.items():
        print(f"  {number}. {name.title()}")

    selected = {}
    for number, name in INTERESTS.items():
        while True:
            try:
                value = float(input(f"  {name.title()} (0-5): "))
                if 0 <= value <= 5:
                    selected[name] = value
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
    print("\n[AI Agent] Analyzing student profile using Holland RIASEC model & Skill Matrix...")
    recommendations = recommend_careers(
        interests=interests,
        skills=skills,
        top_n=5,
    )

    print("\n" + "=" * 70)
    print("TOP CAREER RECOMMENDATIONS (Ranked by Match Score & Supervised ML Job Zone)")
    print("=" * 70)
    for index, row in recommendations.iterrows():
        pred_zone = row.get("Predicted Job Zone", "N/A")
        print(
            f"  {index + 1}. {row['Title']:<38} | "
            f"Job Zone: {row['Job Zone']} (ML Pred: {pred_zone}) | "
            f"Match: {row['Match Score (%)']:.1f}%"
        )

    if auto_choice is not None and 1 <= auto_choice <= len(recommendations):
        choice = auto_choice
        print(f"\n[AI Agent] Auto-selected Top Career Match #{choice}: {recommendations.iloc[choice - 1]['Title']}")
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
    parser.add_argument("--demo", action="store_true", help="Run automated agent demo using Persona 1")
    parser.add_argument("--persona", choices=["1", "2", "3"], help="Run preset student persona (1, 2, or 3)")
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

        # Interactive Mode selection
        print("\nSelect an option:")
        print("  1. Interactive Questionnaire (Input custom interests and skills)")
        print("  2. Demo Profile 1: Computer Science & AI Aspirant")
        print("  3. Demo Profile 2: Digital Media & Creative Game Designer")
        print("  4. Demo Profile 3: Healthcare & Biotechnology Researcher")

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
