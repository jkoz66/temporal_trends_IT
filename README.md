# 📘 Temporal Tech Trends

This repository contains the full pipeline used for the **Temporal Tech Trends** project, including  
data collection, preprocessing, annotation merging, and all figures used in the report.

The instructions below explain how to fully reproduce the results after cloning the repository.

---

##  2. Repository Structure

```
code/
    RQ1.py                           → Generates RQ1 plots
    RQ2.py                           → Generates RQ2 plots
    merge_hype_with_context.py       → Generates RQ3 plots
    analyze_hype_annotations.py      → Annotation evaluation (agreement, distributions)
    merge_buzzword_timeseries.py     → Combines OpenAlex + Trends + Wikipedia
    test_openalex.py                 → OpenAlex API collector
    google_trends_download_and_normalize.py → Google Trends collector
    wikipedia.py                     → Wikipedia pageview collector

data/
    data_openalex/                   → OpenAlex raw + processed + annotation context
    data_trends/                     → Google Trends monthly + yearly data
    data_wiki/                       → Wikipedia monthly + yearly pageviews
    buzzword_timeseries_master.csv   → Final dataset used for RQ1–RQ2
    Annotation_results.csv           → Final hype annotations

figures/                             → All generated figures
```

---

##  Collect Raw Data

If you want to re-download all raw data:

### **OpenAlex**
```bash
python code/test_openalex.py
```

### **Google Trends**
```bash
python code/google_trends_download_and_normalize.py
```

### **Wikipedia Pageviews**
```bash
python code/wikipedia.py
```

---

##  Build Master Time Series Dataset

```bash
python code/merge_buzzword_timeseries.py
```

**Output:**
```
data/buzzword_timeseries_master.csv
```

---

##  Reproduce RQ1 Plots

```bash
python code/RQ1.py
```

**Outputs:**
```
figures/RQ1_hypecycles_smallmultiples.png
```

---

##  Reproduce RQ2 Plots

```bash
python code/RQ2.py
```

**Outputs:**
```
figures/RQ2_timeseries_academic_vs_public.png
figures/RQ2_scatter_academic_vs_public.png
figures/RQ2_corr_academic_vs_public.png
```

---

##  Reproduce RQ3 Plots

```bash
python code/merge_hype_with_context.py
```

**Outputs:**
```
data/data_openalex/annotation/hype_with_context.csv
figures/hype_vs_academic_in_year.png
figures/hype_vs_public_interest.png
figures/hype_score_distribution.png
```

---

##  8. All Figures

After running steps **4–7**, the `figures/` directory will contain **all plots used in the report**.

---
