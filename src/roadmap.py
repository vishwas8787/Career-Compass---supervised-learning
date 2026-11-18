from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_profiles.csv"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"

# ==============================================================================
# DOMAIN-SPECIFIC CURRICULUM CATALOGS (NO VENDOR/LANGUAGE VERSIONS OR CODE CODES)
# ==============================================================================

DOMAIN_CURRICULA = {
    "ARTS_AND_DESIGN": {
        "foundational": [
            "Observational Drawing, Perspective & Composition",
            "Color Theory and Visual Principles",
            "Art History & Aesthetic Movements",
            "Studio Materials & Media Techniques",
        ],
        "core": [
            "Studio Painting & Sculptural Techniques",
            "Digital Illustration & Visual Storytelling",
            "Figure Drawing and Anatomical Studies",
            "Printmaking & Mixed Media Expression",
        ],
        "advanced": [
            "Advanced Studio Practice & Conceptual Art",
            "Visual Development & Gallery Curation",
            "Capstone Creative Portfolio & Exhibition",
        ],
        "certifications": [
            "Professional Fine Arts Portfolio Accreditation",
            "Studio Artist Residency Credential",
        ],
    },
    "GAME_DESIGN": {
        "foundational": [
            "Visual Design Fundamentals & Color Theory",
            "Game Mechanics & Interactive Storytelling",
            "2D Asset Creation & Animation",
            "Mathematics for Computer Graphics",
        ],
        "core": [
            "3D Asset Modeling & Texture Mapping",
            "Game Engine Architecture & Interactive Scripting",
            "Level Design & Player Experience",
            "Audio Design & Environmental Foley",
        ],
        "advanced": [
            "Systems Design & Gameplay Balancing",
            "Interactive Physics & Real-Time Simulation",
            "Capstone Playable Game Prototype",
        ],
        "certifications": [
            "Professional Game Design & Engine Certification",
            "Interactive Media Systems Credential",
        ],
    },
    "COMPUTER_AND_IT": {
        "foundational": [
            "Computer Programming Fundamentals",
            "Discrete Mathematics",
            "Digital Logic and Computer Architecture",
            "Linear Algebra and Numerical Methods",
        ],
        "core": [
            "Data Structures and Algorithms",
            "Database Systems & Structured Query Language",
            "Operating Systems & Computer Networking",
            "Probability Theory and Applied Statistics",
        ],
        "advanced": [
            "Machine Learning & Pattern Recognition",
            "Cloud Infrastructure & Distributed Systems",
            "Software Engineering & Agile Methodologies",
        ],
        "certifications": [
            "Cloud Infrastructure Solutions Architect Certification",
            "Applied Data Science & Machine Learning Credential",
        ],
    },
    "HEALTHCARE_AND_MEDICAL": {
        "foundational": [
            "General Biology with Laboratory",
            "Human Anatomy and Physiology",
            "Medical Terminology & Healthcare Ethics",
            "General Chemistry & Laboratory Safety",
        ],
        "core": [
            "General Pathophysiology & Disease Mechanisms",
            "Biochemistry & Cellular Mechanisms",
            "Health Information Systems & Electronic Records",
            "Biostatistics and Research Methodology",
        ],
        "advanced": [
            "Clinical Epidemiology & Evidence-Based Practice",
            "Principles of Pharmacology & Diagnostics",
            "Healthcare Quality Management & Patient Safety",
        ],
        "certifications": [
            "Certified Healthcare Information & Management Credential",
            "Clinical Specialist Certification",
        ],
    },
    "BUSINESS_AND_FINANCE": {
        "foundational": [
            "Principles of Microeconomics and Macroeconomics",
            "Financial Accounting Principles",
            "Business Communication & Professional Ethics",
            "Business Mathematics and Applied Statistics",
        ],
        "core": [
            "Managerial Accounting & Corporate Finance",
            "Quantitative Business Analytics & Forecasting",
            "Operations and Project Management",
            "Organizational Behavior and Leadership",
        ],
        "advanced": [
            "Financial Modeling & Investment Analysis",
            "Strategic Management & Market Analysis",
            "Corporate Governance & Regulatory Compliance",
        ],
        "certifications": [
            "Financial Modeling and Valuation Credential",
            "Project Management Certification (PMP/CAPM)",
        ],
    },
    "ENGINEERING_AND_TECH": {
        "foundational": [
            "Engineering Mechanics & Statics",
            "Computer-Aided Design (CAD) & Drafting",
            "Differential Equations and Applied Calculus",
            "General Physics & Laboratory Techniques",
        ],
        "core": [
            "Materials Science & Engineering Thermodynamics",
            "Circuit Analysis & Instrumentation",
            "Fluid Mechanics and Heat Transfer",
            "Engineering Statistics & Quality Control",
        ],
        "advanced": [
            "Control Systems & Industrial Automation",
            "Systems Engineering & Reliability Analysis",
            "Senior Engineering Capstone Project",
        ],
        "certifications": [
            "Fundamentals of Engineering (FE) Credential",
            "Certified Computer-Aided Design Professional",
        ],
    },
    "NATURAL_SCIENCE_AND_RESEARCH": {
        "foundational": [
            "General Chemistry & Biology with Laboratory",
            "Calculus and Analytical Geometry",
            "Scientific Inquiry & Laboratory Safety",
            "Introduction to Scientific Computing",
        ],
        "core": [
            "Organic Chemistry & Molecular Biology",
            "Applied Statistics for Scientific Research",
            "Instrumentation & Analytical Methods",
            "Experimental Design and Hypothesis Testing",
        ],
        "advanced": [
            "Advanced Data Analysis in Scientific Research",
            "Computational Modeling & Simulation",
            "Scientific Thesis and Literature Review",
        ],
        "certifications": [
            "Certified Laboratory Research Professional",
            "Scientific Data Analysis Credential",
        ],
    },
    "EDUCATION_AND_HUMAN_SERVICES": {
        "foundational": [
            "Foundations of Education & Learning Theories",
            "Introduction to General Psychology",
            "Interpersonal Communication & Group Dynamics",
            "Child and Adolescent Development",
        ],
        "core": [
            "Instructional Design & Curriculum Development",
            "Educational Technology & Digital Learning Tools",
            "Assessment, Evaluation & Learning Analytics",
            "Counseling Theories and Practical Ethics",
        ],
        "advanced": [
            "Differentiated Instruction & Universal Design",
            "Educational Leadership and Program Evaluation",
            "Practicum Teaching & Supervised Fieldwork",
        ],
        "certifications": [
            "State Teaching & Instructional Credential",
            "Certified Instructional Designer Credential",
        ],
    },
    "GENERAL_APPLIED": {
        "foundational": [
            "Workplace Communication & Professional Fundamentals",
            "Practical Quantitative Reasoning",
            "Occupational Health, Safety & Standards",
            "Applied Digital Tools & Information Literacy",
        ],
        "core": [
            "Technical Operations & Practical Equipment Handling",
            "Customer Service Principles & Problem Resolution",
            "Quality Assurance and Process Documentation",
            "Supervised Technical Apprenticeship",
        ],
        "advanced": [
            "Advanced Technical Specialization & Field Operations",
            "Team Coordination and Workflow Management",
            "Applied Practical Capstone Assessment",
        ],
        "certifications": [
            "Occupational Safety and Health Administration (OSHA) Credential",
            "Vocational Specialist Credential",
        ],
    },
}


def detect_domain(occupation_code, title):
    """Accurately identify career domain using SOC Major Group and title keywords."""
    soc_str = str(occupation_code).strip()
    prefix = soc_str[:2] if len(soc_str) >= 2 else ""
    title_cf = str(title).casefold()

    # Title keyword overrides
    if any(w in title_cf for w in ("video game", "game designer")):
        return "GAME_DESIGN"

    if any(w in title_cf for w in (
        "artist", "painter", "sculptor", "illustrat", "craft", "actor",
        "dancer", "music", "model", "makeup", "animator", "photograph", "curator"
    )):
        return "ARTS_AND_DESIGN"

    if any(w in title_cf for w in (
        "data scientist", "software", "programmer", "developer",
        "database", "computer", "web developer", "network", "cybersecurity"
    )):
        return "COMPUTER_AND_IT"

    if any(w in title_cf for w in (
        "nurse", "physician", "doctor", "therapist", "dentist", "surgeon",
        "pharmacist", "clinical", "medical", "cytotechnologist", "health informatics"
    )):
        return "HEALTHCARE_AND_MEDICAL"

    if any(w in title_cf for w in (
        "engineer", "engineering", "architect", "robotics", "aerospace"
    )):
        return "ENGINEERING_AND_TECH"

    if any(w in title_cf for w in (
        "accountant", "financial", "auditor", "economist", "banker",
        "marketing", "business", "manager", "operations research"
    )):
        return "BUSINESS_AND_FINANCE"

    if any(w in title_cf for w in (
        "biologist", "chemist", "physicist", "microbiologist", "epidemiologist",
        "astronomer", "geologist", "scientist"
    )):
        return "NATURAL_SCIENCE_AND_RESEARCH"

    # SOC Major Group fallback
    soc_group_mapping = {
        "11": "BUSINESS_AND_FINANCE",
        "13": "BUSINESS_AND_FINANCE",
        "15": "COMPUTER_AND_IT",
        "17": "ENGINEERING_AND_TECH",
        "19": "NATURAL_SCIENCE_AND_RESEARCH",
        "21": "EDUCATION_AND_HUMAN_SERVICES",
        "23": "EDUCATION_AND_HUMAN_SERVICES",
        "25": "EDUCATION_AND_HUMAN_SERVICES",
        "27": "ARTS_AND_DESIGN",
        "29": "HEALTHCARE_AND_MEDICAL",
        "31": "HEALTHCARE_AND_MEDICAL",
        "39": "ARTS_AND_DESIGN",
        "41": "BUSINESS_AND_FINANCE",
    }

    return soc_group_mapping.get(prefix, "GENERAL_APPLIED")


def load_profiles():
    """Load occupation profiles containing requirements and descriptions."""
    if PROFILES_FILE.exists():
        return pd.read_csv(PROFILES_FILE)
    if FEATURES_FILE.exists():
        return pd.read_csv(FEATURES_FILE)

    raise FileNotFoundError("Occupation profiles dataset not found.")


def find_occupation(profiles, occupation_code=None, title=None):
    """Find an occupation by its code or title."""
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
        raise ValueError(f"Occupation '{occupation_code or title}' not found.")

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
            raise ValueError(f"Skill rating for '{name}' must be between 0 and 5.")

        validated[name.strip()] = rating

    return validated


def get_skill_gaps(occupation, student_skills, domain="GENERAL_APPLIED"):
    """Rank skill gaps and provide domain-appropriate actionable exercises."""
    gaps = []
    occupation_columns = {
        str(col).casefold(): col for col in occupation.index
    }

    # Domain-aware skill development exercises
    domain_exercises = {
        "ARTS_AND_DESIGN": {
            "Speaking": "Deliver an articulate 5-minute artist talk explaining the concept and inspiration behind your work.",
            "Writing": "Author a professional artist statement and exhibition catalog notes describing your creative process.",
            "Critical Thinking": "Critique classical and contemporary artwork to evaluate aesthetic choices and visual rhetoric.",
            "Active Learning": "Study and replicate techniques from established studio masters; document creative insights.",
            "Active Listening": "Participate in formal studio critique sessions, gather peer feedback, and apply revisions.",
            "Mathematics": "Apply geometric proportions, perspective scaling, and composition grids in studio work.",
            "Reading Comprehension": "Read historical art analyses and exhibition reviews to expand contextual visual literacy.",
            "Science": "Explore material science of pigments, mediums, and curing processes used in studio art.",
            "Monitoring": "Track iterative stages of each artwork from preliminary sketch to final framed presentation.",
            "Learning Strategies": "Maintain a daily visual sketchbook to practice deliberate mark-making and composition.",
        },
        "COMPUTER_AND_IT": {
            "Speaking": "Deliver an articulate technical presentation explaining system architecture to peer engineers.",
            "Writing": "Author clear technical documentation, API specifications, and architectural design memos.",
            "Critical Thinking": "Deconstruct complex algorithmic problems and analyze time and space complexity tradeoffs.",
            "Active Learning": "Build a proof-of-concept prototype when exploring a new framework or technology stack.",
            "Active Listening": "Conduct requirements-gathering sessions with users and translate feedback into tickets.",
            "Mathematics": "Solve applied linear algebra and discrete probability problems using computational scripts.",
            "Reading Comprehension": "Read technical whitepapers and system RFCs to evaluate architectural design choices.",
            "Science": "Formulate testable performance hypotheses, measure latencies, and profile bottlenecks.",
            "Monitoring": "Implement automated testing and application telemetry to monitor operational reliability.",
            "Learning Strategies": "Conduct weekly code reviews and spaced repetition on algorithmic problem patterns.",
        },
        "HEALTHCARE_AND_MEDICAL": {
            "Speaking": "Practice patient communication and multidisciplinary case presentations during rounds.",
            "Writing": "Document structured clinical notes, diagnostic evaluations, and evidence-based summaries.",
            "Critical Thinking": "Evaluate differential diagnoses and weigh clinical evidence against diagnostic guidelines.",
            "Active Learning": "Review recent peer-reviewed medical publications to remain updated on clinical protocols.",
            "Active Listening": "Engage in patient interviews with empathetic listening to elicit complete medical histories.",
            "Mathematics": "Calculate clinical dosage adjustments, statistical efficacy ratios, and patient metrics.",
            "Reading Comprehension": "Analyze peer-reviewed medical literature and clinical trial protocols.",
            "Science": "Investigate underlying pathophysiological mechanisms and pharmacological pathways.",
            "Monitoring": "Track patient vital trajectories and clinical milestones against established care pathways.",
            "Learning Strategies": "Use structured case-based learning to review clinical diagnostic algorithms.",
        },
    }

    fallback_exercises = {
        "Speaking": "Deliver a structured 5-minute presentation explaining core domain topics to colleagues.",
        "Writing": "Draft clear professional summaries and documentation outlining processes and results.",
        "Critical Thinking": "Analyze underlying assumptions, evaluate empirical evidence, and identify risks.",
        "Active Learning": "Identify an emerging methodology in your field and complete a guided practical study.",
        "Active Listening": "Engage in structured stakeholder discussions, confirm understanding, and summarize action items.",
        "Mathematics": "Apply quantitative reasoning and numerical problem solving to practical workplace tasks.",
        "Reading Comprehension": "Review professional publications and synthesize key insights into concise briefs.",
        "Science": "Formulate hypotheses, track controlled variables, and interpret experimental data.",
        "Monitoring": "Set key milestone targets and evaluate ongoing progress against defined quality standards.",
        "Learning Strategies": "Apply active recall and structured practice routines to master essential techniques.",
    }

    chosen_exercises = domain_exercises.get(domain, fallback_exercises)

    for skill_name, student_rating in student_skills.items():
        key = f"skill_{skill_name}".casefold()
        col = occupation_columns.get(key)
        if col is None:
            continue

        requirement = pd.to_numeric(occupation[col], errors="coerce")
        if pd.isna(requirement):
            continue

        requirement = float(requirement)
        gap = max(requirement - student_rating, 0.0)

        exercise = chosen_exercises.get(
            skill_name, fallback_exercises.get(skill_name, f"Engage in structured practice addressing {skill_name}.")
        )

        if gap > 0:
            gaps.append({
                "skill": skill_name,
                "current": student_rating,
                "importance": requirement,
                "gap": round(gap, 2),
                "priority": round(gap * requirement, 3),
                "actionable_exercise": exercise,
            })

    gaps.sort(key=lambda item: item["priority"], reverse=True)
    return gaps


def get_knowledge_areas(occupation, top_n=5):
    """Identify the occupation's highest-rated knowledge areas."""
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


def get_course_roadmap(domain, job_zone=4):
    """Retrieve curriculum stages appropriate for the career domain and preparation level."""
    curriculum = DOMAIN_CURRICULA.get(domain, DOMAIN_CURRICULA["GENERAL_APPLIED"])

    stage_1 = curriculum["foundational"]
    stage_2 = curriculum["core"]
    stage_3 = curriculum["advanced"] if float(job_zone or 3) >= 3.0 else []
    certs = curriculum["certifications"]

    return {
        "stage_1_foundational": stage_1,
        "stage_2_core": stage_2,
        "stage_3_advanced": stage_3,
        "certifications": certs,
    }


def get_job_zone_guidance(job_zone):
    """Explain typical preparation associated with the Job Zone."""
    guidance = {
        1: [
            "Preparation Level: Job Zone 1 (Little or No Preparation)",
            "Required Education: High school diploma or equivalent qualification.",
            "Experience Needed: Little or no previous work experience required.",
            "Training: Short demonstration to a few months of on-the-job training.",
        ],
        2: [
            "Preparation Level: Job Zone 2 (Some Preparation Needed)",
            "Required Education: High school diploma or certificate program.",
            "Experience Needed: Some previous work-related experience or vocational coursework.",
            "Training: A few months to one year of on-the-job training and practical apprenticeship.",
        ],
        3: [
            "Preparation Level: Job Zone 3 (Medium Preparation Needed)",
            "Required Education: Vocational school degree, Associate's Degree, or specialized diploma.",
            "Experience Needed: Previous work-related skill or experience usually needed (1-2 years).",
            "Training: 1 to 2 years of vocational training, studio apprenticeship, or practical coursework.",
        ],
        4: [
            "Preparation Level: Job Zone 4 (Considerable Preparation Needed)",
            "Required Education: Most occupations require a 4-year Bachelor's Degree.",
            "Experience Needed: A considerable amount of work-related skill, knowledge, or experience (2-4 years).",
            "Training: Several years of applied project experience, coursework, and portfolio building.",
        ],
        5: [
            "Preparation Level: Job Zone 5 (Extensive Preparation Needed)",
            "Required Education: Graduate degree typically required (Master's Degree, Doctorate, or Professional Degree).",
            "Experience Needed: Extensive skill, knowledge, and experience (5+ years of specialized background).",
            "Training: Advanced graduate study, research fellowships, or clinical internships.",
        ],
    }

    try:
        zone = int(float(job_zone))
    except (TypeError, ValueError):
        zone = 4

    return guidance.get(zone, ["Research the education and training required for this occupation."])


def get_project_suggestions(title, domain):
    """Generate practical portfolio projects aligned with the occupation and domain."""
    if domain == "ARTS_AND_DESIGN":
        return [
            "Themed Artwork Collection: Create a series of 4–6 cohesive original works (paintings, sculptures, or illustrations) exploring a clear aesthetic theme.",
            "Studio Medium & Technique Study: Experiment with varied traditional or digital media and document the material process and iterations.",
            "Curated Public Portfolio Showcase: Assemble a comprehensive portfolio with high-resolution captures, artist statements, and process documentation.",
        ]

    if domain == "GAME_DESIGN":
        return [
            "Playable Gameplay Prototype: Build a working 2D or 3D prototype demonstrating core game mechanics, camera controls, and interaction loops.",
            "Game Design Document (GDD): Author a comprehensive design document detailing game systems, character balancing, narrative arcs, and UI flows.",
            "Interactive Level Design & Playtesting: Design an interactive environment, conduct structured playtesting sessions, and iterate based on player feedback.",
        ]

    if domain == "COMPUTER_AND_IT":
        return [
            "End-to-End Application: Design and deploy a complete functional application with database storage, robust error handling, and clean architecture.",
            "Data Pipeline or Analytical Model: Ingest and clean a messy public dataset, implement predictive modeling or analytical queries, and document results.",
            "Open Source or Utility Library: Develop and publish a reusable utility tool or library complete with unit tests and clear documentation.",
        ]

    if domain == "HEALTHCARE_AND_MEDICAL":
        return [
            "Clinical Case Analysis: Review an evidence-based clinical case scenario and document differential diagnosis pathways and therapeutic guidelines.",
            "Healthcare Workflow or Data Audit: Analyze health system workflows or data pipelines to identify quality improvements and compliance measures.",
            "Public Health Intervention Proposal: Draft a structured preventive health program targeting community health risks with measurable outcome metrics.",
        ]

    if domain == "BUSINESS_AND_FINANCE":
        return [
            "Financial Valuation & Sensitivity Model: Construct an equity valuation model and scenario analysis evaluating capital expenditure options.",
            "Market Analysis & Growth Strategy: Conduct competitive research on an industry sector and present strategic recommendations with projected returns.",
            "Operations & Workflow Optimization: Map out an enterprise workflow, identify bottlenecks, and propose an optimized implementation plan.",
        ]

    if domain == "ENGINEERING_AND_TECH":
        return [
            "Mechanical or Electrical CAD Prototype: Design a 3D component or circuit schematic and evaluate it against structural or electrical specifications.",
            "Simulation & Performance Analysis: Run computational finite element analysis (FEA) or circuit simulations and document performance margins.",
            "Engineering Design Document: Author a formal engineering report covering design assumptions, test observations, and safety factor calculations.",
        ]

    if domain == "NATURAL_SCIENCE_AND_RESEARCH":
        return [
            "Empirical Scientific Investigation: Design an experiment with controlled variables, collect quantitative observations, and conduct statistical tests.",
            "Systematic Literature Review: Review and synthesize findings from peer-reviewed scientific literature addressing a focused hypothesis.",
            "Reproducible Scientific Report: Prepare a formal paper formatted with Abstract, Methods, Results, and Discussion with reproducible analysis scripts.",
        ]

    if domain == "EDUCATION_AND_HUMAN_SERVICES":
        return [
            "Comprehensive Curriculum Unit: Design a multi-week instructional plan with clear learning objectives, activities, and formative assessments.",
            "Interactive Learning Activity: Develop an engaging educational simulation or interactive module designed for diverse learner needs.",
            "Program Evaluation & Rubric: Construct an evaluation framework measuring educational outcomes and student engagement metrics.",
        ]

    return [
        f"Domain Analysis & Standards Review: Research modern methodologies and practices used by leading {title} professionals.",
        f"Applied Practical Portfolio Piece: Complete an applied portfolio artifact demonstrating primary skills required in {title}.",
        "Professional Showcase: Document your technical process, outcomes, and reflections in a professional portfolio presentation.",
    ]


def get_career_milestones(job_zone):
    """Generate chronological milestones for transitioning into the target career."""
    try:
        zone = int(float(job_zone))
    except (TypeError, ValueError):
        zone = 4

    if zone <= 2:
        return [
            "Phase 1 (Months 1-3): Foundational coursework and core workplace competencies.",
            "Phase 2 (Months 4-6): Hands-on practical training, technical credentialing, and apprenticeship applications.",
            "Phase 3 (Months 7-12): Entry-level industry placement and supervised practical experience.",
        ]
    elif zone == 3:
        return [
            "Phase 1 (Months 1-6): Complete vocational coursework and foundational studio/technical competencies.",
            "Phase 2 (Months 7-12): Earn relevant vocational certification and build hands-on portfolio artifacts.",
            "Phase 3 (Months 13-18): Secure an industry internship, residency, or technician role to refine on-the-job mastery.",
        ]
    elif zone == 4:
        return [
            "Phase 1 (Year 1): Complete foundational coursework in relevant core disciplines.",
            "Phase 2 (Year 2): Develop intermediate competencies, complete major portfolio projects, and publish work.",
            "Phase 3 (Year 3): Pursue industry internships, participate in exhibitions/competitions, and earn specialized credentials.",
            "Phase 4 (Year 4): Capstone project execution, professional networking, and entry-level career placement.",
        ]
    else:  # Zone 5
        return [
            "Phase 1 (Years 1-3): Complete undergraduate degree in relevant foundational disciplines.",
            "Phase 2 (Years 4-5): Pursue graduate/master's studies, specialized research, and peer-reviewed work.",
            "Phase 3 (Years 6+): Advanced residency, doctoral research, or industry fellowship leading to specialized senior practice.",
        ]


def generate_roadmap(occupation, student_skills, predicted_job_zone=None):
    """
    Build a personalized, domain-accurate, multi-step roadmap:
    (Courses -> Skills -> Projects -> Career)
    """
    student_skills = validate_student_skills(student_skills)

    title = str(occupation.get("Title", "Target Career"))
    code = str(occupation.get("O*NET-SOC Code", "N/A"))
    description = str(occupation.get("Description", ""))
    job_zone = occupation.get("Job Zone", 4)

    # Detect exact career domain to prevent mismatched curricula
    domain = detect_domain(code, title)

    skill_gaps = get_skill_gaps(occupation, student_skills, domain=domain)
    knowledge_areas = get_knowledge_areas(occupation)
    course_roadmap = get_course_roadmap(domain, job_zone=job_zone)
    projects = get_project_suggestions(title, domain)
    career_milestones = get_career_milestones(job_zone)
    education_guidance = get_job_zone_guidance(job_zone)

    return {
        "title": title,
        "code": code,
        "domain": domain,
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

    if roadmap.get("description"):
        print(f"\nDescription:\n  {roadmap['description']}")

    # STEP 1: COURSES
    print("\n" + "-" * 70)
    print("STEP 1: COURSEWORK ROADMAP (COURSES)")
    print("-" * 70)
    cr = roadmap["course_roadmap"]
    print("  Stage 1: Foundational Coursework (Core Principles):")
    for course in cr["stage_1_foundational"]:
        print(f"    * {course}")

    print("\n  Stage 2: Intermediate & Major Coursework:")
    for course in cr["stage_2_core"]:
        print(f"    * {course}")

    if cr.get("stage_3_advanced"):
        print("\n  Stage 3: Advanced Specialization & Capstone:")
        for course in cr["stage_3_advanced"]:
            print(f"    * {course}")

    if cr.get("certifications"):
        print("\n  Recommended Professional Credentials & Certifications:")
        for cert in cr["certifications"]:
            print(f"    * {cert}")

    # STEP 2: SKILLS
    print("\n" + "-" * 70)
    print("STEP 2: SKILL DEVELOPMENT & GAP REMEDIATION (SKILLS)")
    print("-" * 70)
    gaps = roadmap["gaps"]
    if gaps:
        print("  Priority Skill Gaps Identified (Your Rating vs Benchmark Requirement):")
        for item in gaps[:5]:
            print(
                f"    * {item['skill']}: Current={item['current']:.1f}/5.0 | "
                f"Required={item['importance']:.2f}/5.0 | Gap={item['gap']:.2f}"
            )
            print(f"      Action: {item['actionable_exercise']}")
    else:
        print("  All assessed skills meet or exceed target occupational benchmarks.")

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
    occupation = find_occupation(
        profiles, title="Fine Artists, Including Painters, Sculptors, and Illustrators"
    )

    student_skills = {
        "Speaking": 2.0,
        "Critical Thinking": 2.5,
        "Active Listening": 3.0,
        "Mathematics": 2.0,
        "Active Learning": 3.0,
    }

    roadmap = generate_roadmap(occupation, student_skills, predicted_job_zone=3.0)
    print_roadmap(roadmap)


if __name__ == "__main__":
    main()
