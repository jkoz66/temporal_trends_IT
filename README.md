Temporal Tech Trends
================================================

This repository contains the full pipeline used for the *Temporal Tech Trends* project, including data collection, preprocessing, annotation merging, and all plots used in the report.

The instructions below explain how to reproduce all results from scratch after cloning the repository.

------------------------------------------------
Collect raw data (optional)
------------------------------------------------

If you want to re-download raw data:

OpenAlex:
    python code/test_openalex.py

Google Trends:
    python code/google_trends_download_and_normalize.py

Wikipedia:
    python code/wikipedia.py

------------------------------------------------
Build master time series dataset
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
7. Reproduce RQ3 plots
------------------------------------------------

python code/merge_hype_with_context.py

Outputs:
    data/data_openalex/annotation/hype_with_context.csv
    figures/hype_vs_academic_in_year.png
    figures/hype_vs_public_interest.png
    figures/hype_score_distribution.png

the figures/ directory will contain all plots included in the report.
