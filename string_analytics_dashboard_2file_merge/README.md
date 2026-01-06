# String Analytics Dashboard (2-file merge)

## What it does
- Upload **DATA** (e.g. `202510.csv`) containing: `date, name, label, Performance`
- Upload **STRING CONFIG** containing: `label, Capacity, String Tilt, String Azimuth, Plant, Roof, Inverter`
- App merges them by column **label** (left join: DATA -> CONFIG)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- If a label in DATA is missing in CONFIG, the config columns will be blank for that row.
- The app accepts CSV/XLSX/Parquet for convenience.
