#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
#%%
data = {
    'name': ['T-5', 'T-6','T-7','T-8'],
    'macro': ['Intact', 'Intact', 'Crushed', 'Crushed'],
    'micro': ['Intact','Crushed', 'Intact','Crushed'],
    'control': ['CK', 'CK', 'CK', 'CK'],
    'CO2_mean': [12.72,10.61833,9.32833,9.94667,],
    'CO2_std': [0.3,0.3425,0.2175,0.16251,],
    'N2O_mean': [46.35333,26.225,17.59,29.705,],
    'N2O_std': [1.3575,1.195,0.605,0.58,],
    'CH4_mean': [47.88333,0,19.295,0,],
    'CH4_std': [1.8025,0,1.015,0,],
    'CO2_a': [11.01737839,10.84317324,11.08871981,10.91451466,],
    'N2O_a': [37.36552121,37.24402429,27.57588294,27.45438601,],
    'CH4_a': [39.19046701,39.32938445,33.33641813,33.47533557,],
}
df = pd.DataFrame(data)
for col in ['CO2_mean', 'N2O_mean', 'CH4_mean', 
            'CO2_std', 'N2O_std', 'CH4_std',
            'CO2_a', 'N2O_a', 'CH4_a']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

gases = ['CO2', 'N2O', 'CH4']

def get_sig_label(p_val):
    if np.isnan(p_val):
        return ' '
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    return 'n.s.'

for gas in gases:
    mean_col = f'{gas}_mean'
    std_col = f'{gas}_std'
    a_col = f'{gas}_a'

    df[f'{gas}_pct_diff'] = ((df[mean_col] - df[a_col]) / df[a_col]) * 100

    p_vals = []
    for _, row in df.iterrows():
        diff = row[mean_col] - row[a_col]
        sd = row[std_col]

        if sd == 0 or np.isnan(sd):
            p_vals.append(np.nan)
        else:
            sim_diff = np.array([diff - sd, diff, diff + sd])
            _, p_val = stats.ttest_1samp(sim_diff, 0)
            p_vals.append(p_val)

    df[f'{gas}_p_val'] = p_vals
    df[f'{gas}_sig'] = [get_sig_label(p) for p in p_vals]



# %%
size1 = 22
fontdict = {'weight': 'bold','size':size1,'family':'SimHei'}
mpl.rcParams.update(
    {
        'text.usetex': False,                  
        #'font.family': 'sans-serif',          
        'font.sans-serif': ['Arial'],          
        'mathtext.fontset': 'stix',            
        'font.size': 14                   
    }
)
fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'wspace': 0.3})
x = np.arange(len(df['name']))
colors = ['#4C72B0', '#55A868', '#C44E52']  # CO2, N2O, CH4的颜色
ylabels = ['CO$_2$ Flux (mg C kg soil d$^{-1}$)', 'N$_2$O Flux (μg N kg soil d$^{-1}$)', 'CH$_4$ Flux (μg C kg soil d$^{-1}$)']
gas_label = ['CO$_2$', 'N$_2$O','CH$_4$']
for i, gas in enumerate(gases):
    ax = axes[i]

    bars = ax.bar(
        x,
        df[f'{gas}_mean'],
        yerr=df[f'{gas}_std'],
        capsize=5,
        color=colors[i],
        edgecolor='black',
        alpha=0.8,
        width=0.6,
        label='Measured (Mean ± STD)',
    )

    ax.plot(
        x,
        df[f'{gas}_a'],
        color='black',
        marker='o',
        linewidth=2,
        markersize=7,
        label=f'{gas}_a',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df['name'], fontsize=18, fontweight='normal')
    ax.set_title(gas_label[i], fontsize=18, fontweight='normal')
    ax.set_ylabel(ylabels[i], fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    # ax.legend(loc='upper right', frameon=True)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
observed_legend = Patch(facecolor='gray', edgecolor='black', label='Observed')
predicted_legend = Line2D([0], [0], marker='o', color='black', linestyle='-',
                          linewidth=2, markersize=8, markerfacecolor='black', 
                          markeredgecolor='black', label='Predicted')

handles, labels =ax.get_legend_handles_labels()
ax.legend(handles=[observed_legend, predicted_legend], bbox_to_anchor=(1.05, -0.08),ncol = 2,
            loc='upper right', fontsize=18, frameon=False, framealpha=0.9)    
fig.suptitle("A: Comparison of observed and predicted GHG emissions", 
             fontsize=18, fontweight='bold', x= 0.3, y=1.05)

pos = axes[0].get_position() 
ax_heatmap = fig.add_axes([pos.x0+0.15,pos.y0-1.05,pos.width*2,pos.height])

heatmap_data = df[[f'{gas}_pct_diff' for gas in gases]].T
heatmap_data.columns = df['name']
heatmap_data.index = gases

annot_matrix = np.empty(heatmap_data.shape, dtype=object)
for i, gas in enumerate(gases):
    for j in range(len(df)):
        pct = df.loc[j, f'{gas}_pct_diff']
        sig = df.loc[j, f'{gas}_sig']
        annot_matrix[i, j] = f'{pct:.1f}%\n({sig})'
vmax = 50
vmin = -vmax

im = sns.heatmap(
    heatmap_data,
    annot=annot_matrix,
    fmt='',
    cmap='vlag',
    center=0,
    vmax=vmax,vmin = vmin,
    ax=ax_heatmap,
    annot_kws={'size': 12, 'weight': 'bold'},
    cbar=False,
    linewidths=1,
    linecolor='white',
    square=False)
cm = im.collections[0]

pos2 = ax_heatmap.get_position() 
cbar_pos = fig.add_axes([pos2.x1+0.02,pos2.y0,pos2.width*0.03,pos2.height])
cbar = fig.colorbar(cm, cax=cbar_pos, orientation='vertical',extend='both')
cbar.set_label('Percentage Difference (%)', fontsize=16, fontweight='normal')
cbar.ax.tick_params(labelsize=16)

ax_heatmap.tick_params(axis='both', labelsize=18)

for spine in ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color('black')
ax_heatmap.set_xticklabels(['T-5', 'T-6', 'T-7', 'T-8'], rotation=0, fontsize=18)
ax_heatmap.set_yticklabels(['CO$_2$', 'N$_2$O','CH$_4$'], rotation=0, fontsize=18)
ax_heatmap.set_xlabel('', fontsize=18)

    
ax_heatmap.set_title("B: Heatmap of the percentage deviation from additive predictions.",
                          fontsize=18, fontweight='bold', pad = 18, x= 0.12)

plt.tight_layout()
plt.savefig('./fig4.png', dpi=300, bbox_inches='tight')
