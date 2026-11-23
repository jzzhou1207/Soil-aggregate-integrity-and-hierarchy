#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl
#%%
data = {
    'name': ['T-1', 'T-2','T-3','T-4'],
    'macro': ['Intact', 'Crushed', 'N.S.', 'N.S.'],
    'micro': ['N.S.','N.S.', 'Intact','Crushed'],
    'control': ['CK', 'CK', 'CK', 'CK'],
    'CO2_mean': [10.59333,10.67167,15.34,13.39],
    'CO2_std': [0.2275,0.35501,0.285,0.33],
    'N2O_mean': [37.55,26.8,35.485,34.125],
    'N2O_std': [1.485,1.235,1.245,0.8],
    'CH4_mean': [39.63,33.20167,34.71,36.265],
    'CH4_std': [1.79,1.39,1.275,1.7],
}
df = pd.DataFrame(data)

n_obs = 3 
gases = ['CO2', 'N2O', 'CH4']
levels = ['macro', 'micro']

ratio_data = []
ttest_results = {}

for level in levels:
    ttest_results[level] = {}
    
    intact_row = df[df[level] == 'Intact'].iloc[0]
    crushed_row = df[df[level] == 'Crushed'].iloc[0]
    
    for gas in gases:
        m_intact = intact_row[f'{gas}_mean']
        s_intact = intact_row[f'{gas}_std']
        
        m_crushed = crushed_row[f'{gas}_mean']
        s_crushed = crushed_row[f'{gas}_std']
        
        ratio = (m_crushed - m_intact) / m_intact * 100
        ratio_data.append({'level': level, 'gas': gas, 'ratio': ratio})
        
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=m_crushed, std1=s_crushed, nobs1=n_obs,
            mean2=m_intact, std2=s_intact, nobs2=n_obs
        )
        ttest_results[level][gas] = p_val

df_ratio = pd.DataFrame(ratio_data)

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

fig = plt.figure(figsize=(15, 12))
gs = plt.GridSpec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.30)

colors = ['#5BB3E4', '#009D73', '#E59F24']
gas_labels = ['CO$_2$', 'N$_2$O', 'CH$_4$']
ylabels = ['CO$_2$ Flux (mg C kg soil d$^{-1}$)', 'N$_2$O Flux (μg N kg soil d$^{-1}$)', 'CH$_4$ Flux (μg C kg soil d$^{-1}$)']


for i, level in enumerate(levels):
    for j, gas in enumerate(gases):
        ax = fig.add_subplot(gs[i, j])
        
        intact_row = df[df[level] == 'Intact'].iloc[0]
        crushed_row = df[df[level] == 'Crushed'].iloc[0]
        
        x = [0, 1]
        means = [intact_row[f'{gas}_mean'], crushed_row[f'{gas}_mean']]
        stds = [intact_row[f'{gas}_std'], crushed_row[f'{gas}_std']]
        
        ax.errorbar(x, means, yerr=stds, fmt='o-', color=colors[j], 
                    ecolor=colors[j], elinewidth=1.5, capsize=4, markersize=8, alpha=0.9)
        ax.set_title(gas_labels[j], fontsize=14, fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels(['Intact', 'Crushed'], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)

        ax.set_ylabel(ylabels[j], fontsize=12)
        if (i == 0) & (j==0) :
            ax.text(-0.35, 0.5, 'Macro-aggregate', transform=ax.transAxes,
                    fontsize=16, fontweight='normal', rotation=90,
                    va='center', ha='center')
        if (i == 1) & (j==0) :
            ax.text(-0.35, 0.5, 'Micro-aggregate', transform=ax.transAxes,
                    fontsize=16, fontweight='normal', rotation=90,
                    va='center', ha='center')      
        gas_means = df[f'{gas}_mean']
        gas_stds = df[f'{gas}_std']
        y_min = (gas_means - gas_stds).min() * 0.95
        y_max = (gas_means + gas_stds).max() * 1.05
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.3, 1.3)
fig.suptitle("A: Effect of aggregate disruption on the GHG emissions", 
             fontsize=16, fontweight='bold', x= 0.3, y=0.95)

pos = gs[2, 0].get_position(fig)

ax_macro_ratio = fig.add_axes([pos.x0, pos.y0, pos.width * 1.65, pos.height])
ax_micro_ratio = fig.add_axes([pos.x0 + pos.width * 1.9, pos.y0, pos.width * 1.65, pos.height])

x_pos = np.array([0, 1, 2])

macro_ratios = df_ratio[df_ratio['level'] == 'macro']['ratio'].values
micro_ratios = df_ratio[df_ratio['level'] == 'micro']['ratio'].values

ax_macro_ratio.bar(x_pos, macro_ratios, color=colors, alpha=0.8, edgecolor='black', width=0.45)
ax_macro_ratio.set_ylabel('Response ratio (%)', fontsize=14)
ax_macro_ratio.set_xlabel('Macro-aggregate', fontsize=12, fontweight='bold')
ax_macro_ratio.set_xticks(x_pos)
ax_macro_ratio.set_xticklabels(gas_labels, fontsize=12)
ax_macro_ratio.grid(True, alpha=0.3, axis='y', linestyle='--')
ax_macro_ratio.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

ax_micro_ratio.bar(x_pos, micro_ratios, color=colors, alpha=0.8, edgecolor='black', width=0.45)
ax_micro_ratio.set_xlabel('Micro-aggregate', fontsize=12, fontweight='bold')
ax_micro_ratio.set_xticks(x_pos)
ax_micro_ratio.set_xticklabels(gas_labels, fontsize=12)
ax_micro_ratio.grid(True, alpha=0.3, axis='y', linestyle='--')
ax_micro_ratio.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

for ax, ratios, level in zip([ax_macro_ratio, ax_micro_ratio], [macro_ratios, micro_ratios], ['macro', 'micro']):
    for i, v in enumerate(ratios):
        gas = gases[i]
        p_value = ttest_results[level][gas]
        
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
            
        label_text = f'{v:.2f}% {sig}'
        
        offset = 2 if v >= 0 else -2
        va = 'bottom' if v >= 0 else 'top'
        
        ax.text(x_pos[i], v + offset, label_text, 
                ha='center', va=va, fontsize=10, fontweight='bold')

ax_macro_ratio.set_ylim(-40, 20)
ax_micro_ratio.set_ylim(-20, 20)

ax_macro_ratio.set_title("B: Response ratio of aggregate disruption on GHG emissions",
                          fontsize=16, fontweight='bold', pad=12, x=0.6)

plt.show()

# %%
# plot
fig = plt.figure(figsize=(15, 10))
gs = plt.GridSpec(2, 3, height_ratios=[0.8, 1], hspace=0.35, wspace=0.30)

for j, gas in enumerate(gases):
    ax = fig.add_subplot(gs[0, j])
    
    macro_intact = df[df['macro'] == 'Intact'].iloc[0]
    macro_crushed = df[df['macro'] == 'Crushed'].iloc[0]
    micro_intact = df[df['micro'] == 'Intact'].iloc[0]
    micro_crushed = df[df['micro'] == 'Crushed'].iloc[0]
    
    means = [
        macro_intact[f'{gas}_mean'], macro_crushed[f'{gas}_mean'],
        micro_intact[f'{gas}_mean'], micro_crushed[f'{gas}_mean']
    ]
    stds = [
        macro_intact[f'{gas}_std'], macro_crushed[f'{gas}_std'],
        micro_intact[f'{gas}_std'], micro_crushed[f'{gas}_std']
    ]
    
    x = [0, 1, 2.2, 3.2]
    

    ax.errorbar(x[:2], means[:2], yerr=stds[:2], fmt='o-', color=colors[j],
                ecolor=colors[j], elinewidth=1.5, capsize=4, markersize=8, alpha=0.9, label='Macro')

    ax.errorbar(x[2:], means[2:], yerr=stds[2:], fmt='o-', color=colors[j],
                ecolor=colors[j], elinewidth=1.5, capsize=4, markersize=8, alpha=0.9, label='Micro')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Intact', 'Crushed', 'Intact', 'Crushed'], fontsize=14)
    ax.set_title(gas_labels[j], fontsize=14, fontweight='normal')
    ax.set_ylabel(ylabels[j], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    gas_means = df[f'{gas}_mean']
    gas_stds = df[f'{gas}_std']
    y_min = (gas_means - gas_stds).min() * 0.95
    y_max = (gas_means + gas_stds).max() * 1.05
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.4, 3.6)
    
    ax.axvline(x=1.6, color='gray', linestyle=':', alpha=0.6)
    ax.text(0.5, y_max - (y_max - y_min) * 0.05, 'Macro', ha='center', va='top', fontsize=14, fontweight='normal', color='#333333')
    ax.text(2.7, y_max - (y_max - y_min) * 0.05, 'Micro', ha='center', va='top', fontsize=14, fontweight='normal', color='#333333')
fig.suptitle("A: Effect of aggregate disruption on the GHG emissions", 
             fontsize=15, fontweight='bold', x= 0.32, y=0.95)

pos = gs[1, 0].get_position(fig)

ax_macro_ratio = fig.add_axes([pos.x0, pos.y0, pos.width * 1.65, pos.height])
ax_micro_ratio = fig.add_axes([pos.x0 + pos.width * 1.9, pos.y0, pos.width * 1.65, pos.height])

x_pos = np.array([0, 1, 2])

macro_ratios = df_ratio[df_ratio['level'] == 'macro']['ratio'].values
micro_ratios = df_ratio[df_ratio['level'] == 'micro']['ratio'].values

ax_macro_ratio.bar(x_pos, macro_ratios, color=colors, alpha=0.8, edgecolor='black', width=0.45)
ax_macro_ratio.set_ylabel(r'RR$_\mathrm{disruption} \, (\%)$', fontsize=14)
ax_macro_ratio.set_xlabel('Macro-aggregate', fontsize=12, fontweight='bold')
ax_macro_ratio.set_xticks(x_pos)
ax_macro_ratio.set_xticklabels(gas_labels, fontsize=12)
ax_macro_ratio.grid(True, alpha=0.3, axis='y', linestyle='--')
ax_macro_ratio.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

ax_micro_ratio.bar(x_pos, micro_ratios, color=colors, alpha=0.8, edgecolor='black', width=0.45)
ax_micro_ratio.set_xticks(x_pos)
ax_micro_ratio.set_xticklabels(gas_labels, fontsize=12)
ax_micro_ratio.set_xlabel('Micro-aggregate', fontsize=12, fontweight='bold')
ax_micro_ratio.grid(True, alpha=0.3, axis='y', linestyle='--')
ax_micro_ratio.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

for ax, ratios, level in zip([ax_macro_ratio, ax_micro_ratio], [macro_ratios, micro_ratios], ['macro', 'micro']):
    for i, v in enumerate(ratios):
        gas = gases[i]
        p_value = ttest_results[level][gas]
        
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
            
        label_text = f'{v:.2f}% {sig}'
        
        offset = 2 if v >= 0 else -2
        va = 'bottom' if v >= 0 else 'top'
        
        ax.text(x_pos[i], v + offset, label_text, 
                ha='center', va=va, fontsize=10, fontweight='bold')

ax_macro_ratio.set_ylim(-40, 20)
ax_micro_ratio.set_ylim(-20, 20)

ax_macro_ratio.set_title("B: Response ratio of aggregate disruption on GHG emissions",
                          fontsize=15, fontweight='bold', pad=12, x=0.6)

plt.tight_layout()
plt.savefig('./fig3.png', dpi=300, bbox_inches='tight')

# %%
