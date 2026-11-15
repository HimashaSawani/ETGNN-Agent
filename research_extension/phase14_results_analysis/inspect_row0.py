import pandas as pd
df_test = pd.read_csv("research_extension/phase13_advanced_investigation/PHASE13_TEST_METRICS.csv")
print(df_test.iloc[0].to_dict())

