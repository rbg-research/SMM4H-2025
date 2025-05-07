# SMM4H-2025
Detection of insomnia in clinical notes

## 1. Setup
```commandline
git clone https://github.com/rbg-research/SMM4H-2025.git
cd SMM4H-2025
pip install -r requirements.txt
```
### Path configurations
Set the correct path to your dataset in **config.py** before running the script below.
```commandline
DATASET_PATH = "data/testing.csv"
```
## 2. Data Preparation
Has to follow the instruction shared by [SMM4H organizers](https://github.com/guilopgar/SMM4H-HeaRD-2025-Task-4-Insomnia).

### Training Data Collation
```commandline
python text_mimic_notes.py --note_ids_path [path_to_note_ids_txt] --mimic_path [path_to_mimic_csv_directory] --output_path data/train.csv
```

### Validation Data Collation
```commandline
python text_mimic_notes.py --note_ids_path [path_to_note_ids_txt] --mimic_path [path_to_mimic_csv_directory] --output_path data/validation.csv
```

### Test Data Collation
```commandline
python text_mimic_notes.py --note_ids_path [path_to_note_ids_txt] --mimic_path [path_to_mimic_csv_directory] --output_path data/test.csv
```


## 3. Notebooks

| S.No | Filename                         | Notebook Description                        | Link                                        |
|------|----------------------------------|---------------------------------------------|---------------------------------------------|
|   1  | validation.ipynb                 | To run model with the validation dataset    | [Open](notebooks/validation.ipynb)  |
|   2  | testing.ipynb                    | To run the model with test dataset          | [Open](notebooks/testing.ipynb) |
|   3  | benchmark.ipynb                  | To Benchmark the model knowledge on subtasks data | [Open](notebooks/Benchmark.ipynb) | 

