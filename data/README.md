# Data

## UCI Online Retail

- Source: https://archive.ics.uci.edu/dataset/352/online+retail
- Citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5BW33
- License: CC BY 4.0
- Coverage: UK online-retail transactions from 2010-12-01 through 2011-12-09
- Downloaded file: `raw/online_retail.zip`
- Extracted file: `raw/Online Retail.xlsx`
- ZIP SHA-256: `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`
- XLSX SHA-256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`

The raw files are local artifacts and are ignored by Git. To reproduce the
processed forecasting data, run:

```bash
uv run prepare-data
```

The preparation step removes cancellations and non-positive quantities/prices,
keeps the 100 highest-volume products by default, aggregates quantity by day,
and fills missing product-days with zero sales.
