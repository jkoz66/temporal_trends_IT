Temporal Tech Trends — Reproduction Guide
================================================

This repository contains the full pipeline used for the *Temporal Tech Trends* project, including data collection, preprocessing, annotation merging, and all plots used in the report.

The instructions below explain how to reproduce all results from scratch after cloning the repository.

------------------------------------------------
1. Install dependencies
------------------------------------------------

pip install -r requirements.txt

If no requirements.txt exists, required packages include: pandas, matplotlib, seaborn, requests, tqdm.

------------------------------------------------
2. Repository structure
------------------------------------------------

code/
    RQ1.py                          → Generates RQ1 plots
    RQ2.py                          → Generates RQ2 plots
    merge_hype_with_context.py      → Generates RQ3 plots
    analyze_hype_annotations.py     → Annotation evaluation (alpha, score distribution)
    merge_buzzword_timeseries.py    → Combines OpenAlex + Trends + Wikipedia
    test_openalex.py                → OpenAlex API collector
    test_trends.py                  → Google Trends collector
    wikipedia.py                    → Wikipedia pageview collector

data/
    data_openalex/                  → OpenAlex raw + processed + annotation context
    data_trends/                    → Google Trends monthly + yearly data
    data_wiki/                      → Wikipedia monthly + yearly pageviews
    buzzword_timeseries_master.csv  → Final merged dataset used for RQ1–RQ2
    Annotation_results.csv          → Final hype annotations

figures/                            → All generated images

README.md                           → This file

------------------------------------------------
3. Collect raw data (optional)
------------------------------------------------

If you want to re-download raw data:

OpenAlex:
    python code/test_openalex.py

Google Trends:
    python code/google_trends_download_and_normalize.py

Wikipedia:
    python code/wikipedia.py

------------------------------------------------
4. Build master time series dataset
------------------------------------------------

python code/merge_buzzword_timeseries.py

Outputs:
    data/buzzword_timeseries_master.csv

------------------------------------------------
5. Reproduce RQ1 plots
------------------------------------------------

python code/RQ1.py

Outputs:
    figures/RQ1_hypecycles_smallmultiples.png

------------------------------------------------
6. Reproduce RQ2 plots
------------------------------------------------

python code/RQ2.py

Outputs:
    figures/RQ2_timeseries_academic_vs_public.png
    figures/RQ2_scatter_academic_vs_public.png
    figures/RQ2_corr_academic_vs_public.png

------------------------------------------------
7. Reproduce RQ3 plots (hype vs academic/public)
------------------------------------------------

python code/merge_hype_with_context.py

Outputs:
    data/data_openalex/annotation/hype_with_context.csv
    figures/hype_vs_academic_in_year.png
    figures/hype_vs_public_interest.png
    figures/hype_score_distribution.png

------------------------------------------------
8. Reproduce annotation metrics
------------------------------------------------

python code/analyze_hype_annotations.py

Produces:
    figures/hype_score_distribution.png
    figures/hype_vs_citations*.png (optional)

------------------------------------------------
9. Done
------------------------------------------------

After running steps 4–7, the figures/ directory will contain all plots included in the report.
