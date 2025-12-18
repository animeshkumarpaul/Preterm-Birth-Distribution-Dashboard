# Preterm Birth Distribution Dashboard (Alberta Pregnancy Cohort)

A Streamlit dashboard that summarizes prescription medication use during pregnancy and associated preterm birth outcomes,
stratified by ATC drug groups.

## Related publication
This dashboard is part of the point-of-care decision support tool described in:

Paul, A.K., Kalmady, S.V., Greiner, R. et al. **Developing point-of-care tools to inform decisions regarding prescription medication use in pregnancy.** *npj Women’s Health* 3, 43 (2025).

- Article page: https://www.nature.com/articles/s44294-025-00093-9  
- DOI: https://doi.org/10.1038/s44294-025-00093-9

If you use or adapt this codebase, please cite the paper above.

## What is included in this repo
- `streamlit_app.py`: Streamlit application entrypoint.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container build for Cloud Run / generic Docker deployment.

## What is NOT included (private data)
The app expects two pickle files that are not suitable for public release:
1. `pregnancy_aggregated_data_all_ATC_CODES.pkl`
2. `mapping_atc_codes.pkl`

Place them locally under `data/` (recommended), or point to them via environment variables.

### Environment variables
- `AGGREGATED_PICKLE_PATH` (default: `data/pregnancy_aggregated_data_all_ATC_CODES.pkl`)
- `ATC_MAPPING_PICKLE_PATH` (default: `data/mapping_atc_codes.pkl`)
- `MIN_CELL_COUNT` (default: `20`) — disclosure-control threshold for showing a drug group.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Run with Docker
```bash
docker build -t preterm-dashboard .
docker run --rm -p 8080:8080 \
  -e AGGREGATED_PICKLE_PATH=/app/data/pregnancy_aggregated_data_all_ATC_CODES.pkl \
  -e ATC_MAPPING_PICKLE_PATH=/app/data/mapping_atc_codes.pkl \
  -v $(pwd)/data:/app/data \
  preterm-dashboard
```

Then open: `http://localhost:8080`

## Citation
### Paper
Paul, A.K., Kalmady, S.V., Greiner, R. et al. Developing point-of-care tools to inform decisions regarding prescription medication use in pregnancy. *npj Women’s Health* 3, 43 (2025). https://doi.org/10.1038/s44294-025-00093-9

### BibTeX
```bibtex
@article{Paul2025POC,
  title   = {Developing point-of-care tools to inform decisions regarding prescription medication use in pregnancy},
  author  = {Paul, A. K. and Kalmady, S. V. and Greiner, R. and others},
  journal = {npj Women’s Health},
  volume  = {3},
  pages   = {43},
  year    = {2025},
  doi     = {10.1038/s44294-025-00093-9},
}
```

## Disclaimer
This dashboard is intended for exploratory, descriptive summaries. It does not establish causation.
