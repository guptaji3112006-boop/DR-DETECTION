import pandas as pd

prov_file = "external_data/idrid/iqa_provenance_manifest"
test_file = "validation_sim/results/test_manifest.csv"

prov_df = pd.read_csv(prov_file)
test_df = pd.read_csv(test_file)

print(f"Provenance CSV total rows: {len(prov_df)}")
train_records = prov_df[prov_df['split'] == 'training']
test_records = prov_df[prov_df['split'] == 'testing']

print(f"Provenance CSV Training records: {len(train_records)}")
print(f"Provenance CSV Testing records: {len(test_records)}")

# Only Training IDRiD_001.jpg through IDRiD_005.jpg are marked as used for threshold tuning.
tuning_used = prov_df[prov_df['used_for_iqa_threshold_tuning'] == True]
print(f"Records marked used_for_iqa_threshold_tuning: {len(tuning_used)}")
for idx, row in tuning_used.iterrows():
    print(f" - {row['image_name']} (split: {row['split']}, hash: {row['sha256']})")

tuning_hashes = set(tuning_used['sha256'])
test_hashes = set(test_df['sha256'])

tuning_in_test = tuning_hashes.intersection(test_hashes)
print(f"Tuning hashes present in test_manifest: {len(tuning_in_test)}")

# All 103 Testing hashes match our previously checked test manifest.
prov_test_hashes = set(test_records['sha256'])
match_test = prov_test_hashes == test_hashes
print(f"All 103 Testing hashes match test_manifest exactly: {match_test}")

# Training IDRiD_118.jpg and Testing IDRiD_064.jpg share recorded SHA-256
train_118 = train_records[train_records['image_name'] == 'IDRiD_118.jpg']['sha256'].values
test_064 = test_records[test_records['image_name'] == 'IDRiD_064.jpg']['sha256'].values

if len(train_118) > 0 and len(test_064) > 0:
    print(f"Train IDRiD_118.jpg hash: {train_118[0]}")
    print(f"Test IDRiD_064.jpg hash: {test_064[0]}")
    print(f"Do they match? {train_118[0] == test_064[0]}")
else:
    print("Could not find one of the specific files to compare.")

