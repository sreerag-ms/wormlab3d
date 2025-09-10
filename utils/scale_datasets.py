import pandas as pd

input_file = 'data/cleaned_dataset.csv'
output_file = 'data/cleaned_dataset_scaled.csv'

df = pd.read_csv(input_file)

df['x_head'] = df['x_head'] * 2
df['y_head'] = df['y_head'] * 2
df['x_tail'] = df['x_tail'] * 2
df['y_tail'] = df['y_tail'] * 2

df.to_csv(output_file, index=False)
print(f"Doubled coordinates saved to: {output_file}")
