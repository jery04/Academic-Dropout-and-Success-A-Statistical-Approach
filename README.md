🎓📊 Statistics Project 2025–2026 — Academic Dropout and Success 📚✨🧪💻📈
---

**Team members:**
- Alex Moreno Rodríguez — C312
- Jery Rodríguez Fernández — C312

**Project summary 🧠🔬**
A statistical approach to study academic dropout and success among higher-education students using real data to identify patterns, risk factors, and build predictive models 🧠📈. The project combines descriptive analysis, visualization, clustering, and predictive modeling to understand which features are associated with retention and academic achievement. Results and figures are saved under the `outputs/` folder.

**Data source 🔗📂**
- Kaggle — Predict Students’ Dropout and Academic Success
- Observations: ~4,424 students 🔢
- Type: structured, multivariate data 🗄️
- Link: https://www.kaggle.com/datasets/thedevastator/higher-educationpredictors-of-student-retention 🔗

**Research questions ❓🔬**
1. Do students who graduate show higher performance values than students who drop out?
2. How are different academic performance profiles structured, and how do they relate to socioeconomic conditions and the probability of persistence or dropout?
3. Can dropout or success be predicted using statistical and classification techniques?

**Project structure 🗂️🛠️ (only structure shown)**

```
Project-Statistics-2025-2026/
|
├─ dataset/
│  ├─ dataset.csv          # Raw data (~4,424 rows)
│  └─ README.md            # Notes about the dataset and variables
├─ digital presentation (ppt)/
│  └─ presentación.pptx    # Presentation slides
├─ mi_entorno/             # Python virtual environment (kept in repo)
│  ├─ bin/
│  ├─ lib64/
│  ├─ pyvenv.cfg
│  └─ share/
├─ notebook/
│  └─ statistical_analysis.ipynb  # Full exploratory and statistical analysis
├─ outputs/
│  ├─ clustering_analysis/
│  │  ├─ conclusiones_clustering.txt
│  │  └─ figures/
│  ├─ data_visualization/
│  │  ├─ boxplots/
│  │  ├─ histograms/
│  ├─ descriptive_stats/
│  │  ├─ descriptive_stats_summary.csv
│  │  └─ ... (more stats files)
│  ├─ performance_analysis/
│  │  └─ ...
│  ├─ prediction_results/
│  │  └─ ...
│  ├─ prepared_data/
│  │  └─ ...
├─ scripts/
│  ├─ clustering_analysis.py
│  ├─ data_preparation.py
│  ├─ data_visualization.py
│  ├─ descriptive_stats.py
│  ├─ Map_var.py
│  ├─ performance_analysis.py
│  └─ predicting_success.py
├─ requirements.txt        # Project dependencies
└─ README.md               # This file
```

**Scripts (brief descriptions — max 3 sentences each) 🧩🛠️**
- `clustering_analysis.py`: 🔬 Performs clustering on the prepared dataset to identify student groups with similar academic and socioeconomic profiles. Produces cluster visualizations and saves cluster summaries and figures to `outputs/clustering_analysis/`.
- `data_preparation.py`: 🧹 Loads the raw `dataset/dataset.csv`, cleans missing or inconsistent values, encodes categorical variables, and saves the processed dataset into `outputs/prepared_data/` for downstream analysis.
- `data_visualization.py`: 📈 Generates exploratory plots (histograms, boxplots, etc.) to inspect distributions and relationships between variables. Figures are exported to `outputs/data_visualization/`.
- `descriptive_stats.py`: 📊 Computes descriptive statistics and summary tables for key variables and writes the aggregated summaries (CSV and text) to `outputs/descriptive_stats/`.
- `Map_var.py`: 🗺️ Defines variable mappings, label encodings, and helper functions used to standardize and document variable transformations across scripts. Exports mapping artifacts when needed for reproducibility.
- `performance_analysis.py`: 📐 Analyzes academic performance measures and compares groups (e.g., graduates vs. dropouts), producing tables and plots that support the research questions. Results are saved under `outputs/performance_analysis/`.
- `predicting_success.py`: 🤖 Trains and evaluates classification models to predict dropout or academic success, reporting performance metrics and saving models and prediction outputs to `outputs/prediction_results/`.
