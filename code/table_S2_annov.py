# %%
import pandas as pd
import numpy as np
import io
import statsmodels.api as sm
from statsmodels.formula.api import ols
# %%
data_str = """Treatment\tMacroIntact\tmicroIntact\tcontrol\tIncubation temperature\tGWP-Mean\tGWP-STD
T-1\tIntact\tN.S.\tCK\t9.4℃\t56.37784\t1.051588492
T-2\tCrushed\tN.S.\tCK\t9.4℃\t51.82191679\t1.406285712
T-3\tN.S.\tIntact\tCK\t9.4℃\t72.71929167\t1.174478591
T-4\tN.S.\tCrushed\tCK\t9.4℃\t65.04183167\t1.259218678
T-5\tIntact\tIntact\tCK\t9.4℃\t68.24937845\t1.246339688
T-6\tIntact\tCrushed\tCK\t9.4℃\t50.18440167\t1.356441119
T-7\tCrushed\tIntact\tCK\t9.4℃\t42.44460667\t0.83946711
T-8\tCrushed\tCrushed\tCK\t9.4℃\t49.21456833\t0.645734039
T-1\tIntact\tN.S.\tOTC\t9.4℃\t44.61797857\t0.915394116
T-2\tCrushed\tN.S.\tOTC\t9.4℃\t52.20131321\t0.717389604
T-3\tN.S.\tIntact\tOTC\t9.4℃\t66.67390655\t1.359431451
T-4\tN.S.\tCrushed\tOTC\t9.4℃\t53.31758667\t0.984408063
T-5\tIntact\tIntact\tOTC\t9.4℃\t58.66866643\t0.640944944
T-6\tIntact\tCrushed\tOTC\t9.4℃\t62.78394833\t1.007200137
T-7\tCrushed\tIntact\tOTC\t9.4℃\t54.24761833\t0.889819393
T-8\tCrushed\tCrushed\tOTC\t9.4℃\t56.45517167\t0.929902164
T-1\tIntact\tN.S.\tCK\t19.4℃\t75.65515833\t1.273764158
T-2\tCrushed\tN.S.\tCK\t19.4℃\t69.4289031\t1.357493615
T-3\tN.S.\tIntact\tCK\t19.4℃\t77.75832167\t1.759607415
T-4\tN.S.\tCrushed\tCK\t19.4℃\t70.64584\t1.361531186
T-5\tIntact\tIntact\tCK\t19.4℃\t73.09339667\t1.567503007
T-6\tIntact\tCrushed\tCK\t19.4℃\t124.0456983\t1.855988715
T-7\tCrushed\tIntact\tCK\t19.4℃\t89.31924833\t1.831663948
T-8\tCrushed\tCrushed\tCK\t19.4℃\t67.14981167\t1.369202211
T-1\tIntact\tN.S.\tOTC\t19.4℃\t86.850745\t1.808255631
T-2\tCrushed\tN.S.\tOTC\t19.4℃\t83.40990167\t2.004135873
T-3\tN.S.\tIntact\tOTC\t19.4℃\t62.46244167\t0.900452576
T-4\tN.S.\tCrushed\tOTC\t19.4℃\t70.25637667\t1.824439782
T-5\tIntact\tIntact\tOTC\t19.4℃\t84.93316\t1.946954325
T-6\tIntact\tCrushed\tOTC\t19.4℃\t94.212135\t2.120363889
T-7\tCrushed\tIntact\tOTC\t19.4℃\t86.29107667\t2.503797043
T-8\tCrushed\tCrushed\tOTC\t19.4℃\t92.076425\t1.765431752"""

df = pd.read_csv(io.StringIO(data_str), sep='\t')
# %%
raw_data_rows = []
for index, row in df.iterrows():
    mean = row['GWP-Mean']
    std = row['GWP-STD']
    values = [mean - std, mean, mean + std]
    temp_val = str(row['Incubation temperature']).replace('℃', '')
    for val in values:
        raw_data_rows.append({
            'Treatment': row['Treatment'],
            'Control': row['control'],
            'Temperature': temp_val,
            'GWP': val
        })
raw_df = pd.DataFrame(raw_data_rows)
treatments = ['T-1', 'T-2', 'T-3', 'T-4', 'T-5', 'T-6', 'T-7', 'T-8']
for t in treatments:
    subset = raw_df[raw_df['Treatment'] == t]
    model = ols('GWP ~ C(Control) + C(Temperature) + C(Control):C(Temperature)', data=subset).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"================ ANOVA Results for {t} ================")
    print(anova_table)
    print("\n")
# %%
