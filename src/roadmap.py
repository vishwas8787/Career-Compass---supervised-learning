"""
Roadmap generation module for CareerCompass.
Generates multi-step roadmaps (Courses -> Skills -> Projects -> Career)
tailored to student ratings and O*NET occupational profiles.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_profiles.csv"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"

# Comprehensive Course Mapping Knowledge Base for O*NET Knowledge Domains
COURSE_CATALOG = {
    "Computers and Electronics": {
        "foundational": [
            "CS 101: Introduction to Computer Science & Programming (Python)",
            "CS 102: Digital Logic and Computer Architecture",
        ],
        "core": [
            "CS 201: Data Structures and Algorithms",
            "CS 205: Database Systems & SQL",
            "CS 210: Operating Systems & Computer Networking",
        ],
        "advanced": [
            "CS 301: Machine Learning & Intelligent Systems",
            "CS 305: Cloud Computing & Distributed Architecture",
            "CS 315: Software Engineering & Agile Development",
        ],
        "certifications": [
            "AWS Certified Solutions Architect / Cloud Practitioner",
            "CompTIA Security+ / Network+",
        ],
    },
    "Mathematics": {
        "foundational": [
            "MATH 101: Differential and Integral Calculus",
            "MATH 105: Discrete Mathematics",
        ],
        "core": [
            "MATH 201: Linear Algebra and Matrix Theory",
            "STAT 202: Probability Theory & Applied Statistics",
        ],
        "advanced": [
            "MATH 301: Numerical Analysis & Optimization Methods",
            "STAT 305: Multivariate Statistical Modeling",
        ],
        "certifications": [
            "Certified Analytics Professional (CAP)",
            "Applied Data Science with Python (Specialization)",
        ],
    },
    "Design": {
        "foundational": [
            "DSGN 101: Fundamentals of Visual Design and Composition",
            "DSGN 105: Human-Centered Design Principles",
        ],
        "core": [
            "DSGN 201: UI/UX Wireframing & Interactive Prototyping (Figma)",
            "DSGN 205: Information Architecture & Usability Testing",
        ],
        "advanced": [
            "DSGN 301: Design Systems, Micro-Interactions & Accessibility",
            "DSGN 305: 3D Asset Modeling & Interactive Motion Graphics",
        ],
        "certifications": [
            "Google UX Design Professional Certificate",
            "Nielsen Norman Group UX Master Certified",
        ],
    },
    "Engineering and Technology": {
        "foundational": [
            "ENGR 101: Engineering Mechanics & Statics",
            "ENGR 105: Computer-Aided Design (CAD) & Rapid Prototyping",
        ],
        "core": [
            "ENGR 201: Circuit Analysis & Embedded Systems",
            "ENGR 205: Materials Science & Engineering Thermodynamics",
        ],
        "advanced": [
            "ENGR 301: Control Systems, Automation & Robotics",
            "ENGR 305: Systems Engineering & Reliability Analysis",
        ],
        "certifications": [
            "Fundamentals of Engineering (FE) Certification",
            "SolidWorks Certified Professional (CSWP)",
        ],
    },
    "Biology": {
        "foundational": [
            "BIO 101: General Biology & Laboratory Techniques",
            "BIO 105: Cellular and Molecular Biology",
        ],
        "core": [
            "BIO 201: Genetics, Heredity & Genomics",
            "BIO 205: Microbiology & Immunology",
        ],
        "advanced": [
            "BIO 301: Bioinformatics & Computational Genomics",
            "BIO 305: Molecular Biotechnology & Genetic Engineering",
        ],
        "certifications": [
            "Bioinformatics Specialization (Coursera/UCSD)",
            "Certified Clinical Research Associate (CCRA)",
        ],
    },
    "Medicine and Dentistry": {
        "foundational": [
            "MED 101: Human Anatomy and Physiology I & II",
            "MED 105: Medical Terminology & Healthcare Ethics",
        ],
        "core": [
            "MED 201: General Pathophysiology & Disease Mechanisms",
            "MED 205: Principles of Pharmacology & Diagnostics",
        ],
        "advanced": [
            "MED 301: Clinical Epidemiology & Evidence-Based Practice",
            "MED 305: Health Informatics & Electronic Health Records",
        ],
        "certifications": [
            "Certified Clinical Medical Assistant (CCMA)",
            "Certified Healthcare Information & Management Systems (CPHIMS)",
        ],
    },
    "Economics and Accounting": {
        "foundational": [
            "ECON 101: Principles of Microeconomics and Macroeconomics",
            "ACCT 101: Financial Accounting Principles",
        ],
        "core": [
            "ACCT 201: Managerial Accounting & Corporate Finance",
            "ECON 205: Intermediate Microeconomic Theory & Markets",
        ],
        "advanced": [
            "ECON 301: Econometrics & Quantitative Financial Modeling",
            "FIN 305: Investment Analysis & Portfolio Management",
        ],
        "certifications": [
            "Chartered Financial Analyst (CFA) Level 1",
            "Financial Risk Manager (FRM)",
        ],
    },
    "Psychology": {
        "foundational": [
            "PSYC 101: Introduction to Psychological Science",
            "PSYC 105: Developmental Psychology Across the Lifespan",
        ],
        "core": [
            "PSYC 201: Cognitive Psychology & Perception",
            "PSYC 205: Research Methods & Behavioral Statistics",
        ],
        "advanced": [
            "PSYC 301: Human-Computer Interaction & Ergonomics",
            "PSYC 305: Behavioral Economics & Decision Making",
        ],
        "certifications": [
            "Board Certified Behavior Analyst (BCBA)",
            "Applied Ergonomics & Human Factors Certificate",
        ],
    },
    "Communications and Media": {
        "foundational": [
            "COMM 101: Introduction to Mass Communication & Media",
            "COMM 105: Public Speaking and Strategic Rhetoric",
        ],
        "core": [
            "COMM 201: Digital Content Production & Multimedia Storytelling",
            "COMM 205: Technical Writing & Editing for Digital Media",
        ],
        "advanced": [
            "COMM 301: Media Analytics & Audience Insights",
            "COMM 305: Crisis Communication & Public Relations Strategy",
        ],
        "certifications": [
            "HubSpot Digital Marketing & Content Marketing Certification",
            "Google Analytics Individual Qualification (GA4)",
        ],
    },
    "Administration and Management": {
        "foundational": [
            "MGMT 101: Principles of Management & Organizational Behavior",
            "MGMT 105: Business Communication & Collaboration",
        ],
        "core": [
            "MGMT 201: Operations and Project Management",
            "MGMT 205: Agile & Scrum Frameworks",
        ],
        "advanced": [
            "MGMT 301: Product Strategy & Roadmapping",
            "MGMT 305: Strategic Leadership & Change Management",
        ],
        "certifications": [
            "Project Management Professional (PMP) / CAPM",
            "Professional Scrum Master I (PSM I)",
        ],
    },
    "Sales and Marketing": {
        "foundational": [
            "MKTG 101: Principles of Marketing & Consumer Psychology",
            "MKTG 105: Professional Sales & Communication Fundamentals",
        ],
        "core": [
            "MKTG 201: Digital Marketing Strategy, SEO & SEM",
            "MKTG 205: Marketing Analytics & Customer Lifetime Value",
        ],
        "advanced": [
            "MKTG 301: Brand Identity & Growth Marketing",
            "MKTG 305: Enterprise Account Management & CRM Systems",
        ],
        "certifications": [
            "Meta Certified Digital Marketing Associate",
            "Salesforce Certified Administrator",
        ],
    },
    "Fine Arts": {
        "foundational": [
            "ART 101: Observational Drawing, Perspective & Composition",
            "ART 105: Color Theory and Visual Principles",
        ],
        "core": [
            "ART 201: Digital Painting & Concept Art",
            "ART 205: 2D Animation & Storyboarding",
        ],
        "advanced": [
            "ART 301: Visual Development & Art Direction for Entertainment",
            "ART 305: Capstone Creative Portfolio & Exhibition",
        ],
        "certifications": [
            "Adobe Certified Professional (Photoshop / Illustrator)",
            "Unity Certified User: Artist",
        ],
    },
    "Chemistry": {
        "foundational": [
            "CHEM 101: General Chemistry I & II with Laboratory",
            "CHEM 105: Chemical Safety & Quantitative Analysis",
        ],
        "core": [
            "CHEM 201: Organic Chemistry I & II",
            "CHEM 205: Analytical Chemistry & Instrumentation",
        ],
        "advanced": [
            "CHEM 301: Biochemistry & Molecular Mechanisms",
            "CHEM 305: Physical Chemistry & Spectroscopy",
        ],
        "certifications": [
            "ACS Certified Chemist Credential",
            "Occupational Health and Safety Chemical Handling (OSHA)",
        ],
    },
    "Physics": {
        "foundational": [
            "PHYS 101: Mechanics and Thermodynamics (Physics I)",
            "PHYS 105: Electromagnetism and Optics (Physics II)",
        ],
        "core": [
            "PHYS 201: Waves, Optics, and Modern Physics",
            "PHYS 205: Computational Physics & Numerical Simulation",
        ],
        "advanced": [
            "PHYS 301: Quantum Mechanics & Electrodynamics",
            "PHYS 305: Statistical Mechanics & Solid State Physics",
        ],
        "certifications": [
            "Computational Modeling in Physics Certificate",
            "Certified LabVIEW Associate Developer",
        ],
    },
    "Education and Training": {
        "foundational": [
            "EDUC 101: Foundations of Education & Learning Theories",
            "EDUC 105: Educational Psychology & Motivation",
        ],
        "core": [
            "EDUC 201: Instructional Design Principles & Curriculum Planning",
            "EDUC 205: Educational Technology & E-Learning Tools",
        ],
        "advanced": [
            "EDUC 301: Assessment, Evaluation & Learning Analytics",
            "EDUC 305: Differentiated Instruction & Universal Design",
        ],
        "certifications": [
            "Certified Instructional Designer (CID)",
            "State Teaching Licensure / Higher Education Teaching Credential",
        ],
    },
    "Law and Government": {
        "foundational": [
            "PLSC 101: Introduction to Political Science & Legal Systems",
            "PLSC 105: Constitutional Law & Civil Liberties",
        ],
        "core": [
            "LAW 201: Administrative Law, Governance & Public Policy",
            "LAW 205: Intellectual Property & Data Privacy Regulations",
        ],
        "advanced": [
            "LAW 301: International Law & Regulatory Compliance",
            "LAW 305: Legal Research, Writing & Policy Negotiation",
        ],
        "certifications": [
            "Certified Information Privacy Professional (CIPP)",
            "Certified Regulatory Compliance Manager (CRCM)",
        ],
    },
}

# Skill gap specific courses and practical exercises
SKILL_DEVELOPMENT_RESOURCES = {
    "Mathematics": {
        "course": "MATH 100: Quantitative Problem Solving & Applied Calculus",
        "exercise": "Solve 5 applied real-world numerical problems weekly and verify using Python/NumPy.",
    },
    "Critical Thinking": {
        "course": "PHIL 102: Logic, Reasoning & Evidence-Based Argumentation",
        "exercise": "Deconstruct a published technical paper or business case study; map assumptions vs. empirical evidence.",
    },
    "Active Learning": {
        "course": "GENS 101: Self-Directed Inquiry & Knowledge Synthesis",
        "exercise": "Pick one new framework or tool every two weeks; build a minimal proof-of-concept and write a learning summary.",
    },
    "Speaking": {
        "course": "COMM 102: Professional Presentations & Pitching for Technical Topics",
        "exercise": "Deliver a 5-minute recorded presentation on a technical topic; review pacing, slide clarity, and articulation.",
    },
    "Writing": {
        "course": "WRIT 201: Technical and Professional Communication",
        "exercise": "Draft clear GitHub documentation, design memos, and project READMEs with architecture diagrams.",
    },
    "Reading Comprehension": {
        "course": "ENGL 105: Critical Literature Analysis & Research Syntheses",
        "exercise": "Read and summarize peer-reviewed journals or standard specifications into 1-page executive briefs.",
    },
    "Science": {
        "course": "SCIE 101: The Scientific Method, Hypothesis Testing & Experimental Design",
        "exercise": "Formulate a testable hypothesis, record observations with controlled variables, and compute confidence intervals.",
    },
    "Active Listening": {
        "course": "COMM 104: Interpersonal Communication & Stakeholder Empathy",
        "exercise": "Engage in simulated stakeholder interviews, take structured notes, and confirm requirements via active playback.",
    },
    "Monitoring": {
        "course": "MGMT 203: Performance Tracking, KPIs & Quality Assurance",
        "exercise": "Establish weekly project milestones with automated metric dashboards to monitor progress against goals.",
    },
    "Learning Strategies": {
        "course": "EDUC 102: Cognitive Learning Strategies & Habit Formation",
        "exercise": "Apply spaced repetition and active recall (Feynman technique) to review core concepts twice weekly.",
    },
}


def load_profiles():
    """Load occupation profiles containing O*NET requirements."""
    if PROFILES_FILE.exists():
        return pd.read_csv(PROFILES_FILE)
    if FEATURES_FILE.exists():
        return pd.read_csv(FEATURES_FILE)

    raise FileNotFoundError(
        f"Occupation profiles not found at {PROFILES_FILE} or {FEATURES_FILE}."
    )


def find_occupation(profiles, occupation_code=None, title=None):
    """Find an occupation by its O*NET code or title."""
    if occupation_code is not None:
        matches = profiles[
            profiles["O*NET-SOC Code"].astype(str) == str(occupation_code)
        ]
    elif title is not None:
        matches = profiles[
            profiles["Title"].astype(str).str.casefold()
            == str(title).casefold()
        ]
    else:
        raise ValueError("Provide an occupation code or title.")

    if matches.empty:
        raise ValueError(
            f"Occupation not found for code='{occupation_code}', title='{title}'."
        )

    return matches.iloc[0]


def validate_student_skills(student_skills):
    """Validate skill ratings supplied by the student."""
    validated = {}

    for name, rating in (student_skills or {}).items():
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            raise ValueError(f"Skill rating for '{name}' must be numeric.")

        if not 0 <= rating <= 5:
            raise ValueError(
                f"Skill rating for '{name}' must be between 0 and 5."
            )

        validated[name.strip()] = rating

    return validated


def get_skill_gaps(occupation, student_skills):
    """Rank skill gaps using the occupation's actual O*NET requirements."""
    gaps = []
    occupation_columns = {
        str(column).casefold(): column for column in occupation.index
    }

    for skill_name, student_rating in student_skills.items():
        key = f"skill_{skill_name}".casefold()
        column = occupation_columns.get(key)

        if column is None:
            continue

        requirement = pd.to_numeric(occupation[column], errors="coerce")
        if pd.isna(requirement):
            continue

        requirement = float(requirement)
        gap = max(requirement - student_rating, 0.0)

        resource = SKILL_DEVELOPMENT_RESOURCES.get(
            skill_name,
            {
                "course": f"Applied Practice in {skill_name}",
                "exercise": f"Engage in structured practice addressing {skill_name}.",
            },
        )

        if gap > 0:
            gaps.append({
                "skill": skill_name,
                "current": student_rating,
                "importance": requirement,
                "gap": round(gap, 2),
                "priority": round(gap * requirement, 3),
                "recommended_course": resource["course"],
                "actionable_exercise": resource["exercise"],
            })

    gaps.sort(key=lambda item: item["priority"], reverse=True)
    return gaps


def get_knowledge_areas(occupation, top_n=5):
    """Identify the occupation's highest-rated O*NET knowledge areas."""
    areas = []

    for column in occupation.index:
        if not str(column).startswith("knowledge_"):
            continue

        value = pd.to_numeric(occupation[column], errors="coerce")
        if pd.isna(value):
            continue

        name = str(column)[len("knowledge_"):]
        areas.append({
            "name": name,
            "importance": round(float(value), 2),
        })

    areas.sort(key=lambda item: item["importance"], reverse=True)
    return areas[:top_n]


def get_course_roadmap(knowledge_areas, skill_gaps, job_zone=4):
    """Map top knowledge areas and skill gaps into structured course stages."""
    foundational_courses = set()
    core_courses = set()
    advanced_courses = set()
    recommended_certifications = set()

    for area in knowledge_areas:
        name = area["name"]
        catalog_entry = COURSE_CATALOG.get(name)

        if catalog_entry:
            foundational_courses.update(catalog_entry.get("foundational", []))
            core_courses.update(catalog_entry.get("core", []))
            if float(job_zone or 3) >= 3.5:
                advanced_courses.update(catalog_entry.get("advanced", []))
            recommended_certifications.update(
                catalog_entry.get("certifications", [])
            )

    # Add remedial courses for the largest skill gaps
    for gap in skill_gaps[:3]:
        course = gap.get("recommended_course")
        if course:
            foundational_courses.add(course)

    # Fallback if no specific catalog matches were found
    if not foundational_courses:
        foundational_courses.add("General Foundations in Analytical Thinking & Computing")
    if not core_courses:
        core_courses.add("Core Applied Practice in Field Specialization")

    return {
        "stage_1_foundational": sorted(list(foundational_courses))[:4],
        "stage_2_core": sorted(list(core_courses))[:4],
        "stage_3_advanced": sorted(list(advanced_courses))[:4],
        "certifications": sorted(list(recommended_certifications))[:3],
    }


def get_job_zone_guidance(job_zone):
    """Explain the typical preparation associated with an O*NET Job Zone."""
    guidance = {
        1: [
            "Preparation Level: Job Zone 1 (Little or No Preparation)",
            "Required Education: High school diploma or GED equivalent.",
            "Experience Needed: Little or no previous work-related skill, knowledge, or experience.",
            "Training: Short demonstration to a few months of on-the-job training.",
        ],
        2: [
            "Preparation Level: Job Zone 2 (Some Preparation Needed)",
            "Required Education: High school diploma or certificate program.",
            "Experience Needed: Some previous work-related skill, knowledge, or experience.",
            "Training: A few months to one year of on-the-job training and practical apprenticeship.",
        ],
        3: [
            "Preparation Level: Job Zone 3 (Medium Preparation Needed)",
            "Required Education: Vocational school degree, Associate's Degree, or specialized certificate.",
            "Experience Needed: Previous work-related skill, knowledge, or experience usually needed (1-2 years).",
            "Training: 1 to 2 years of vocational training or structured apprenticeship.",
        ],
        4: [
            "Preparation Level: Job Zone 4 (Considerable Preparation Needed)",
            "Required Education: Most occupations require a 4-year Bachelor's Degree (B.S., B.A., B.Tech).",
            "Experience Needed: A considerable amount of work-related skill, knowledge, or experience (2-4 years).",
            "Training: Several years of work-related experience, on-the-job training, and project portfolio.",
        ],
        5: [
            "Preparation Level: Job Zone 5 (Extensive Preparation Needed)",
            "Required Education: Graduate degree typically required (Master's Degree, Ph.D., M.D., or J.D.).",
            "Experience Needed: Extensive skill, knowledge, and experience (5+ years of specialized background).",
            "Training: Graduate study, clinical/research internships, and advanced postdoctoral work.",
        ],
    }

    try:
        zone = int(float(job_zone))
    except (TypeError, ValueError):
        zone = 4

    return guidance.get(
        zone,
        ["Research the education and training commonly required for this occupation."],
    )


def get_project_suggestions(title):
    """Return practical portfolio projects tailored to the occupation."""
    title_cf = str(title).casefold()

    if any(word in title_cf for word in ("data scientist", "statistician", "biostatistician", "quantitative analyst")):
        return [
            "Exploratory Data Analysis & Cleaning: Process a messy public dataset (Kaggle/UCI), identify outliers, and visualize key trends.",
            "Predictive Modeling Pipeline: Implement supervised classification and regression pipelines with cross-validation and hyperparameter tuning.",
            "Interactive Data Dashboard: Deploy an end-to-end interactive dashboard (Streamlit/Dash) showcasing reproducible insights.",
        ]

    if any(word in title_cf for word in ("software developer", "software engineer", "web developer", "computer programmer")):
        return [
            "Full-Stack Web Application: Design and deploy a responsive web app with user authentication, RESTful APIs, and database persistence.",
            "System Utility & Algorithmic Tool: Implement a high-performance CLI tool with automated unit test suites and CI/CD automation.",
            "Open Source Contribution: Submit a pull request or develop a reusable open-source library with comprehensive documentation.",
        ]

    if any(word in title_cf for word in ("game designer", "video game")):
        return [
            "Core Gameplay Prototype: Design a playable 2D/3D prototype in Unity or Unreal Engine focusing on core mechanics and game loops.",
            "Comprehensive Game Design Document (GDD): Write a complete GDD detailing narrative arcs, mechanics, balance matrices, and UI mockups.",
            "Level Design & Playtesting: Build an interactive level, conduct playtesting sessions, and iterate based on quantitative player feedback.",
        ]

    if any(word in title_cf for word in ("research scientist", "microbiologist", "epidemiologist", "biochemist")):
        return [
            "Systematic Literature Synthesis: Review recent scientific studies on a focused research question and synthesize key findings.",
            "Computational Data Experiment: Analyze genomic, biological, or epidemiological data using Python/R Bio-packages.",
            "Peer-Format Research Report: Write a complete paper formatted with Abstract, Methods, Results, Discussion, and Reproducibility code.",
        ]

    if any(word in title_cf for word in ("designer", "graphic", "ux", "ui", "multimedia")):
        return [
            "User Experience Case Study: Conduct user research and wireframing to redesign an existing digital product.",
            "Interactive Design System: Create a reusable component library in Figma complete with typography, accessibility guidelines, and interactions.",
            "High-Fidelity Interactive Prototype: Build a clickable mobile/web prototype and document usability testing results.",
        ]

    if any(word in title_cf for word in ("financial", "accountant", "economist", "auditor")):
        return [
            "Corporate Financial Valuation Model: Build a Discounted Cash Flow (DCF) model and sensitivity analysis for a publicly traded firm.",
            "Macroeconomic Impact Brief: Analyze historical inflation, interest rate, and employment datasets to forecast sectoral impacts.",
            "Portfolio Risk Optimization: Construct an optimal asset allocation portfolio using Modern Portfolio Theory in Python/Excel.",
        ]

    if any(word in title_cf for word in ("nurse", "physician", "doctor", "therapist", "medical", "healthcare")):
        return [
            "Clinical Case Review: Examine a complex clinical scenario using peer-reviewed guidelines and document diagnostic reasoning.",
            "Public Health Intervention Plan: Design an evidence-based preventive health initiative targeting a specific community health risk.",
            "Healthcare Safety & Protocol Audit: Review patient safety protocols and develop an educational checklist for clinical practitioners.",
        ]

    return [
        f"Industry Analysis & Domain Review: Research the core technologies and modern methodologies used by leading {title}s.",
        f"Hands-On Milestone Project: Complete an applied portfolio artifact demonstrating primary skills required in {title}.",
        "Professional Presentation: Document technical choices, results, and limitations in a public portfolio or GitHub repository.",
    ]


def get_career_milestones(title, job_zone):
    """Generate chronological milestones for transitioning into the target career."""
    try:
        zone = int(float(job_zone))
    except (TypeError, ValueError):
        zone = 4

    if zone <= 2:
        return [
            "Phase 1 (Months 1-3): Foundational coursework and essential workplace communication practice.",
            "Phase 2 (Months 4-6): Hands-on training, technical certification, and initial apprenticeship application.",
            "Phase 3 (Months 7-12): Entry-level placement and supervised industry experience.",
        ]
    elif zone == 3:
        return [
            "Phase 1 (Months 1-6): Complete vocational/associate coursework in core occupational competencies.",
            "Phase 2 (Months 7-12): Earn relevant vocational certification and build hands-on project artifacts.",
            "Phase 3 (Months 13-18): Secure an industry internship or technician role and refine on-the-job skills.",
        ]
    elif zone == 4:
        return [
            "Phase 1 (Year 1): Complete foundational math, computing, and domain science coursework.",
            "Phase 2 (Year 2): Develop core technical capabilities, complete 2 major portfolio projects, and publish code.",
            "Phase 3 (Year 3): Pursue industry internships, participate in hackathons/conferences, and earn specialized certifications.",
            "Phase 4 (Year 4): Capstone project execution, professional networking, and entry-level career placement.",
        ]
    else:  # Zone 5
        return [
            "Phase 1 (Years 1-3): Complete rigorous undergraduate degree in relevant foundational disciplines.",
            "Phase 2 (Years 4-5): Pursue graduate/master's studies, specialized research, and peer-reviewed publications.",
            "Phase 3 (Years 6+): Advanced residency, doctoral research, or industry fellowship leading to specialized senior practice.",
        ]


def generate_roadmap(occupation, student_skills, predicted_job_zone=None):
    """
    Build a personalized, multi-step roadmap:
    (Courses -> Skills -> Projects -> Career)
    """
    student_skills = validate_student_skills(student_skills)

    title = str(occupation.get("Title", "Target Career"))
    code = str(occupation.get("O*NET-SOC Code", "N/A"))
    description = str(occupation.get("Description", ""))
    job_zone = occupation.get("Job Zone", 4)

    skill_gaps = get_skill_gaps(occupation, student_skills)
    knowledge_areas = get_knowledge_areas(occupation)
    course_roadmap = get_course_roadmap(knowledge_areas, skill_gaps, job_zone=job_zone)
    projects = get_project_suggestions(title)
    career_milestones = get_career_milestones(title, job_zone)
    education_guidance = get_job_zone_guidance(job_zone)

    return {
        "title": title,
        "code": code,
        "description": description,
        "job_zone": job_zone,
        "predicted_job_zone": predicted_job_zone,
        "education_guidance": education_guidance,
        "course_roadmap": course_roadmap,
        "gaps": skill_gaps,
        "knowledge_areas": knowledge_areas,
        "projects": projects,
        "career_milestones": career_milestones,
    }


def print_roadmap(roadmap):
    """Display the complete multi-step roadmap (Courses -> Skills -> Projects -> Career)."""
    print("\n" + "=" * 70)
    print(f"CAREERCOMPASS ROADMAP: {roadmap['title'].upper()}")
    print("=" * 70)
    print(f"O*NET-SOC Code: {roadmap['code']}")
    print(f"O*NET Job Zone: {roadmap['job_zone']}")
    if roadmap.get("predicted_job_zone") is not None:
        print(f"Supervised ML Predicted Job Zone: {roadmap['predicted_job_zone']}")

    if roadmap.get("description"):
        print(f"\nDescription:\n  {roadmap['description']}")

    # STEP 1: COURSES
    print("\n" + "-" * 70)
    print("STEP 1: COURSEWORK ROADMAP (COURSES)")
    print("-" * 70)
    cr = roadmap["course_roadmap"]
    print("  Stage 1: Foundational Coursework (Prerequisites & Core Concepts):")
    for course in cr["stage_1_foundational"]:
        print(f"    * {course}")

    print("\n  Stage 2: Intermediate & Major Coursework:")
    for course in cr["stage_2_core"]:
        print(f"    * {course}")

    if cr.get("stage_3_advanced"):
        print("\n  Stage 3: Advanced Specialization & Applied Electives:")
        for course in cr["stage_3_advanced"]:
            print(f"    * {course}")

    if cr.get("certifications"):
        print("\n  Recommended Industry Certifications:")
        for cert in cr["certifications"]:
            print(f"    * {cert}")

    # STEP 2: SKILLS
    print("\n" + "-" * 70)
    print("STEP 2: SKILL DEVELOPMENT & GAP REMEDIATION (SKILLS)")
    print("-" * 70)
    gaps = roadmap["gaps"]
    if gaps:
        print("  Priority Skill Gaps Identified (Your Rating vs O*NET Requirement):")
        for item in gaps[:5]:
            print(
                f"    * {item['skill']}: Current={item['current']:.1f}/5.0 | "
                f"Required={item['importance']:.2f}/5.0 | Gap={item['gap']:.2f}"
            )
            print(f"      Action: {item['actionable_exercise']}")
    else:
        print("  All assessed skills meet or exceed the target career requirements.")

    # STEP 3: PRACTICAL PROJECTS
    print("\n" + "-" * 70)
    print("STEP 3: PRACTICAL PROJECTS & PORTFOLIO ARTIFACTS")
    print("-" * 70)
    for index, project in enumerate(roadmap["projects"], 1):
        print(f"  {index}. {project}")

    # STEP 4: CAREER MILESTONES
    print("\n" + "-" * 70)
    print("STEP 4: CAREER PREPARATION & MILESTONES (CAREER)")
    print("-" * 70)
    print("  Educational Requirements:")
    for item in roadmap["education_guidance"]:
        print(f"    - {item}")

    print("\n  Chronological Career Transition Milestones:")
    for milestone in roadmap["career_milestones"]:
        print(f"    - {milestone}")
    print("=" * 70)


def main():
    profiles = load_profiles()
    occupation = find_occupation(profiles, title="Data Scientists")

    student_skills = {
        "Speaking": 2.0,
        "Critical Thinking": 2.5,
        "Active Listening": 3.0,
        "Mathematics": 2.0,
        "Active Learning": 3.0,
    }

    roadmap = generate_roadmap(occupation, student_skills, predicted_job_zone=4.0)
    print_roadmap(roadmap)


if __name__ == "__main__":
    main()
