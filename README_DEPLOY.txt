ONE WAY PICKZ — MLB H+R+RBI FULL PACKAGE v2.1

UPLOAD THE CONTENTS OF THIS FOLDER TO THE ROOT OF YOUR GITHUB REPOSITORY.

Required:
- app.py
- requirements.txt
- data/batter_profiles.csv

Included for reproducibility:
- data/raw/cleaned_batting_stats.csv
- cleaned_batting_stats.csv (root fallback copy)
- railway.json
- OneWayPickz_MLB_HRR_v2_1_Full.py (identical backup copy)

Historical integration:
- Raw rows: 4,502 player-seasons, 2015-2024
- Processed player priors: 1,140
- Duplicate traded-player seasons are consolidated.
- Mojibake/accented names and Baseball-Reference batting-side markers are cleaned.
- Rates use PA weighting with a 3-year recency half-life.
- The historical prior supplements 2025 and current 2026 data; it does not replace live data.

Railway start command:
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
