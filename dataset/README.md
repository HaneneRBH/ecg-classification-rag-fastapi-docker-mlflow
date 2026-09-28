# Dataset

This project uses the **PTB-XL** electrocardiography dataset by Wagner et al.
(Physikalisch-Technische Bundesanstalt, Berlin).

The dataset files are **not included** in this repository — they are large
(~3 GB extracted) and their redistribution should go through the official
channel to preserve proper attribution.

## Download

1. Go to the dataset page: [PhysioNet — PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
   (DOI: [`10.13026/6sec-a640`](https://doi.org/10.13026/6sec-a640))
2. **No account required** — PTB-XL is fully open access (Creative Commons
   Attribution 4.0).
3. Click **"Download the ZIP file"** (~1.7 GB) or use wget:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

4. Extract the archive into this `dataset/` folder (or anywhere on your
   machine — pass the path via `--data-root`).

## Expected final layout

```
dataset/
├── ptbxl_database.csv            # main table: one row per ECG (21 799 rows)
├── scp_statements.csv            # SCP codes → diagnostic super-classes
├── records100/                   # 100 Hz signals (recommended for prototyping)
│   ├── 00000/
│   │   ├── 00001_lr.dat
│   │   ├── 00001_lr.hea
│   │   └── ...
│   ├── 01000/
│   └── ...
├── records500/                   # 500 Hz signals (full resolution)
│   └── ...
└── example_physionet.py          # official minimal example
```

## WFDB file format

Each recording is a pair of files sharing the same name:

- **`.hea`** (header) — a short text file describing the signal: number of
  leads, sampling frequency, number of samples, units, lead names.
- **`.dat`** (data) — the raw signal values (16-bit, 1 μV/LSB resolution).

In Python, the `wfdb` library reads both files together:

```python
import wfdb
record = wfdb.rdrecord("dataset/records100/00000/00001_lr")
signal = record.p_signal  # shape (1000, 12) for 100 Hz, 10 s
```

## Key numbers

| Property                | Value                           |
|-------------------------|---------------------------------|
| ECG recordings          | 21 799                          |
| Patients                | 18 869                          |
| Leads                   | 12 standard                     |
| Duration                | 10 seconds each                 |
| Sampling rates          | 500 Hz (original), 100 Hz (down-sampled) |
| Annotators              | Up to 2 cardiologists per ECG   |
| Diagnostic statements   | 71 SCP-ECG codes                |
| Super-classes           | NORM, MI, STTC, CD, HYP        |
| Recommended split       | Folds 1–8 train, 9 val, 10 test |

## Citation

If you use this dataset, please cite the original paper:

```bibtex
@article{wagner2020ptbxl,
  title     = {{PTB-XL}, a large publicly available electrocardiography dataset},
  author    = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter
               and Kreiseler, Dieter and Lunze, Fatima I. and Samek, Wojciech
               and Schaeffter, Tobias},
  journal   = {Scientific Data},
  volume    = {7},
  number    = {1},
  pages     = {154},
  year      = {2020},
  doi       = {10.1038/s41597-020-0495-6}
}
```

And the PhysioNet resource:

```bibtex
@article{goldberger2000physiobank,
  title     = {{PhysioBank}, {PhysioToolkit}, and {PhysioNet}: Components of a
               New Research Resource for Complex Physiologic Signals},
  author    = {Goldberger, Ary L. and Amaral, Luis A. N. and Glass, Leon and
               Hausdorff, Jeffrey M. and Ivanov, Plamen Ch. and Mark, Roger G.
               and Mietus, Joseph E. and Moody, George B. and Peng, Chung-Kang
               and Stanley, H. Eugene},
  journal   = {Circulation},
  volume    = {101},
  number    = {23},
  pages     = {e215--e220},
  year      = {2000},
  doi       = {10.1161/01.CIR.101.23.e215}
}
```
