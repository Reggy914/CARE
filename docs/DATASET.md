# Dataset

## Scope

The release supports the ELE electrical-current dataset format used by the experiment controller. No raw or processed data is included.

## Raw CSV schema

The preprocessor expects:

- one header row;
- a timestamp in the first column;
- one or more current-series columns after the timestamp;
- numeric or numeric-convertible current values.

Example:

~~~csv
timestamp,MT_001,MT_002
2024-01-01T00:00:00Z,0.42,0.53
2024-01-01T00:15:00Z,0.45,0.51
~~~

Timestamp values may be parseable date strings, Unix seconds, or Unix milliseconds. Invalid timestamps are removed. Rows are sorted chronologically and duplicate timestamps keep their first occurrence.

For each current column, non-numeric values are converted to missing values. The preprocessor interpolates missing values in order and fills boundary gaps. A column is skipped if it has no numeric values or still contains missing values after cleaning.

## Preprocessing

~~~bash
python main.py preprocess \
  --raw-csv /path/to/raw/electricity.csv \
  --output-dir data/processed/ele \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2
~~~

Ratios must be positive and sum to one. The command writes:

- ele_processed.csv: timestamp and cleaned current columns;
- data_profile.json: cleaning counts and source profile;
- columns.json: canonical timestamp and target-column metadata;
- split_info.json: chronological split counts and time ranges.

Review data_profile.json after preprocessing. It records invalid timestamps, removed duplicates, skipped columns, and remaining missing-value counts.

## Split and scaling semantics

The loader uses contiguous chronological regions:

~~~text
earliest                                              latest
|---------------- train ----------------|--- val ---|--- test ---|
~~~

There is no row overlap between regions. Windows are created independently inside each region, so a validation or test window never borrows history from the preceding region.

Standardization is fitted only on raw training target values. The same training mean and standard deviation are applied to validation and test values. Evaluation converts forecasts back to the original target units.

Training windows are shuffled by the DataLoader. Validation, test, evaluation, and inference windows retain chronological order. A sample ID is the global row index where its input window begins.

## Window requirements

Every split must contain at least input_len + pred_len rows. With the default lengths of 96 and 96, each of train, validation, and test must contain at least 192 rows.

The stride controls the distance between adjacent window starts. A larger stride reduces overlap and runtime.

## Data governance

Keep private or licensed data outside version control. The default data directory is ignored by Git. Before sharing a derived dataset, verify its source license, privacy constraints, and redistribution terms.
