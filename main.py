import os
import pandas as pd

import config
from src import load_data, InsomniaClassifier, convert_output_to_json


input_path = os.path.join(config.data_directory, config.data_split + ".csv")

df = load_data(config.DATASET_PATH)
clinical_notes = df['text'].tolist()
classifier = InsomniaClassifier()

classification_results = []
extracted_texts = []

# Process each clinical note with exception handling
for idx, clinical_note in enumerate(clinical_notes):
    print(f"Processing text {idx + 1}/{len(clinical_notes)}: {clinical_note[:100]}...")
    try:
        classification, extracted = classifier.classify(clinical_note)
        classification_results.append(classification)
        extracted_texts.append(extracted)
    except RuntimeError as e:
        print(f"RuntimeError for text at index {idx}: {e}")
        # Append default classifications and empty extracted text on error
        classification_results.append({
            "Definition 1 (Sleep Difficulty)": "no",
            "Definition 2 (Daytime Impairment)": "no",
            "Rule A (Insomnia Diagnosis)": "no",
            "Rule B (Primary Medications)": "no",
            "Rule C (Secondary Medications)": "no",
            "Final Insomnia Status": "no"
        })
        extracted_texts.append({
            "Definition 1 Extracted": "",
            "Definition 2 Extracted": "",
            "Rule A Extracted": "",
            "Rule B Extracted": "",
            "Rule C Extracted": ""
        })
    print("-" * 80)


# Convert results to DataFrames
df_classification = pd.DataFrame(classification_results)
df_extracted = pd.DataFrame(extracted_texts)

# Combine all DataFrames
df_final = pd.concat([df[['text', 'note_id']], df_classification, df_extracted], axis=1)

# Rename columns to match expected names for JSON conversion
df_final = df_final.rename(columns={
    "Definition 1 (Sleep Difficulty)": "Definition 1 Pred",
    "Definition 2 (Daytime Impairment)": "Definition 2 Pred",
    "Rule A (Insomnia Diagnosis)": "Rule A Pred",
    "Rule B (Primary Medications)": "Rule B Pred",
    "Rule C (Secondary Medications)": "Rule C Pred",
    "Final Insomnia Status": "Insomnia Pred",
    "Definition 1 Extracted": "Definition 1 Evidence",
    "Definition 2 Extracted": "Definition 2 Evidence",
    "Rule B Extracted": "Rule B Evidence",
    "Rule C Extracted": "Rule C Evidence"
})


output_csv_dir = os.path.join(config.output_directory, config.data_split)
os.makedirs(output_csv_dir, exist_ok=True)
# Save CSV to results folder
csv_output_path = os.path.join(output_csv_dir, "output.csv")
df_final.to_csv(csv_output_path, index=False)

# Generate JSON outputs
convert_output_to_json(csv_output_path)

