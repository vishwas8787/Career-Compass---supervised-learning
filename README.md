# CareerCompass — Career & Education Advisor

CSE395 Supervised Learning Project | Lovely Professional University (LPU)

CareerCompass recommends careers based on a student's interests and skills, predicts the preparation level needed, and creates a step-by-step roadmap from learning to career.

## How It Works

1. **Career matching:** Uses K-Nearest Neighbors (KNN) to find careers that match the student's interests and skills.
2. **Preparation prediction:** Uses a Random Forest classifier to predict the required Job Zone.
3. **Roadmap generation:** Suggests courses, skills to improve, projects, and career milestones.

The project uses occupational data from the O*NET database.

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

### Web App

```bash
streamlit run app/app.py
```

### Command-Line Demo

```bash
python main.py --demo
```

For interactive mode:

```bash
python main.py
```

### Run Tests

```bash
python -m unittest discover tests
```

### Retrain the Model

```bash
python src/train.py
```

## Sample Roadmaps

Three example roadmaps are available in `outputs/sample_roadmaps/`:

- Technology, AI, and data systems
- Creative media and game design
- Healthcare and biotechnology

## Model Evaluation

Evaluation results are saved in `outputs/`:

- `classification_report.txt` — classification report
- `confusion_matrix.png` — confusion matrix
- `roc_curves.png` — ROC curves
- `evaluation_metrics.json` — evaluation metrics

## Main Folders

- `app/` — Streamlit web application
- `data/` — occupational datasets and processed features
- `models/` — trained model
- `outputs/` — evaluation results and sample roadmaps
- `src/` — data processing, training, recommendations, and roadmap generation
- `tests/` — automated tests

## Academic Context

- **Course:** CSE395 Supervised Learning
- **Institution:** Lovely Professional University (LPU)
- **Dataset:** O*NET Database
