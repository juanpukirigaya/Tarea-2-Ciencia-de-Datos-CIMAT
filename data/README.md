
## Data source

The data came from [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## How to use it?

From the parent directory you just need to run
```bash 
python -m scripts.get_raw_data
```

It will download the zip file from the [url](https://datapub.gfz-potsdam.de/download/10.5880.GFZ.4.3.2023.002sdfsd/2023-001_ISONET-Project-Members_13C_Data.zip) and unzip it then put it in the `data/raw` directory.

Both variables `data_url` and `data_dir` are defined in the `src/config.py`.

## References
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.