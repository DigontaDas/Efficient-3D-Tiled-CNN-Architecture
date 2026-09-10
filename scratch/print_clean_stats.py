import pandas as pd

df = pd.read_csv(r'H:\Thesis_Trainings\results\case_level_model_comparison.csv')
for ds in df['Dataset'].unique():
    print(f"\n==================== {ds} ====================")
    for m in ['DICE', 'HD95', 'PRECISION', 'RECALL', 'CLDICE']:
        sub = df[(df['Dataset'] == ds) & (df['Metric'] == m)]
        print(f"\n--- {m} ---")
        for _, r in sub.iterrows():
            mname = r['Model']
            mean_val = r['Mean']
            std_val = r['Std']
            med_val = r['Median']
            q1 = r['Q1_25pct']
            q3 = r['Q3_75pct']
            min_v = r['Min']
            max_v = r['Max']
            print(f"{mname:20s} | Mean: {mean_val:8.4f} +/- {std_val:7.4f} | Med: {med_val:8.4f} [{q1:7.4f}, {q3:7.4f}] | Min: {min_v:8.4f} | Max: {max_v:8.4f}")
