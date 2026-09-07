# Log Transform
import numpy as np
import pandas as pd

# Sample dataset
data = {'Value': [1, 10, 100, 1000, 10000, 100000, -100,
                  -1, -10, 20, -30, 300000, 200000]}
df = pd.DataFrame(data)

# Apply Log10 and Natural Log (ln)
df['Log10'] = np.log10(df['Value'])
df['NaturalLog'] = np.log(df['Value'])
# Log(x + 1) ensure the data is positive
df['Log10(x+1)'] = np.log(df['Value'] + 1)

print(df)
