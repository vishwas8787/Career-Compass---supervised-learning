
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


def get_interests():
    print("\nRate your interests from 0 to 5.")
    print("0 = Not interested, 5 = Very interested")

    for number, name in INTERESTS.items():
        print(f"{number}. {name.title()}")

    selected = {}

    for number, name in INTERESTS.items():
        while True:
            try:
                value = float(input(f"{name.title()} (0-5): "))

                if 0 <= value <= 5:
                    selected[name] = value
                    break

                print("Enter a number between 0 and 5.")
            except ValueError:
                print("Please enter a valid number.")

    return selected

def get_skills():
    print("\nRate your current skills from 0 to 5.")
    print("0 = No experience, 5 = Very proficient")
    print("Press Enter to skip a skill you haven't assessed.")

    skills = {}

    for name in SKILL_NAMES:
        while True:
            raw_value = input(f"{name} (0-5, Enter to skip): ").strip()

            if raw_value == "":
                break

            try:
                value = float(raw_value)

                if 0 <= value <= 5:
                    skills[name] = value
                    break

                print("Enter a number between 0 and 5, or press Enter to skip.")

            except ValueError:
                print("Please enter a valid number, or press Enter to skip.")

    return skills



def main():
    print("=" * 55)
    print("          CAREERCOMPASS")
    print("   Your Education-to-Career Advisor")
    print("=" * 55)

    try:
        interests = get_interests()
        skills = get_skills()

        print("\nFinding careers that match your profile...")

        recommendations = recommend_careers(
            interests=interests,
            skills=skills,
            top_n=5,
        )

        print("\nTop Career Recommendations")
        print("-" * 55)

        for index, row in recommendations.iterrows():
            print(
                f"{index + 1}. {row['Title']}"
                f" | Job Zone: {row['Job Zone']}"
                f" | Match Score: {row['Match Score (%)']:.2f}%"
            )

        while True:
            try:
                choice = int(
                    input("\nChoose a career (1-5): ")
                )

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
        )

        print_roadmap(roadmap)

        print("\nCareerCompass session complete.")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
