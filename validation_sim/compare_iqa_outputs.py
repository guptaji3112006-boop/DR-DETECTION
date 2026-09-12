import pandas as pd
import sys

def compare(old_csv_path, new_csv_path, output_csv_path):
    old_df = pd.read_csv(old_csv_path)
    new_df = pd.read_csv(new_csv_path)

    merged = pd.merge(
        old_df, new_df, on='sample_id', suffixes=('_old', '_new')
    )
    
    comparison_data = []
    changed_count = 0
    for _, row in merged.iterrows():
        has_changed = (row['quality_score_old'] != row['quality_score_new']) or \
                      (row['uniformity_old'] != row['uniformity_new']) or \
                      (row['status_old'] != row['status_new'])
        
        if has_changed:
            changed_count += 1
            
        comparison_data.append({
            'sample_id': row['sample_id'],
            'old_quality_score': row['quality_score_old'],
            'new_quality_score': row['quality_score_new'],
            'old_uniformity': row['uniformity_old'],
            'new_uniformity': row['uniformity_new'],
            'old_status': row['status_old'],
            'new_status': row['status_new'],
            'old_reason': row['reason_old'],
            'new_reason': row['reason_new'],
            'has_changed': has_changed
        })
        
    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(output_csv_path, index=False)
    
    print(f"Comparison saved to {output_csv_path}")
    print(f"Total changed records: {changed_count}")
    
    if changed_count > 0:
        print("\nChanged Records:")
        changed_df = comp_df[comp_df['has_changed']]
        print(changed_df[['sample_id', 'old_status', 'new_status', 'old_uniformity', 'new_uniformity']])

if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], sys.argv[3])
