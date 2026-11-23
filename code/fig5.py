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


data = """name	macro	micro	control	CO2-Mean	N2O-Mean	CH4-Mean	CO2-STD	N2O-STD	CH4-STD
T-1	Intact	N.S.	CK	10.59333	37.55	39.63	0.2275	1.485	1.79
T-2	Crushed	N.S.	CK	10.67167	26.8	33.20167	0.35501	1.235	1.39
T-3	N.S.	Intact	CK	15.34	35.485	34.71	0.285	1.245	1.275
T-4	N.S.	Crushed	CK	13.39	34.125	36.265	0.33	0.8	1.7
T-5	Intact	Intact	CK	12.72	46.35333	47.88333	0.3	1.3575	1.8025
T-6	Intact	Crushed	CK	10.61833	26.225	0	0.3425	1.195	0
T-7	Crushed	Intact	CK	9.32833	17.59	19.295	0.2175	0.605	1.015
T-8	Crushed	Crushed	CK	9.94667	29.705	0	0.16251	0.58	0
T-1	Intact	N.S.	OTC	8.83833	28.46333	0	0.2175	1.0475	0
T-2	Crushed	N.S.	OTC	9.75833	34.99	39.16833	0.14251	1.14	1.3675
T-3	N.S.	Intact	OTC	11.68	50.4	61.82333	0.225	2.515	1.6025
T-4	N.S.	Crushed	OTC	11.38333	26.99	0	0.2375	1.07	0
T-5	Intact	Intact	OTC	12.075	30.37167	37.895	0.145	0.8225	1.675
T-6	Intact	Crushed	OTC	13.53167	27.985	32.285	0.2425	1.1	0.94
T-7	Crushed	Intact	OTC	11.85167	25.155	0	0.2175	0.92	0
T-8	Crushed	Crushed	OTC	12.49	24.845	0	0.245	0.56	0
"""
lines = data.strip().split('\n')
headers = lines[0].split('\t')
rows = [line.split('\t') for line in lines[1:]]

df = pd.DataFrame(rows, columns=headers)
def get_treatment_type(name):
    if name in ['T-1', 'T-2', 'T-3', 'T-4']:
        return 'Single'
    elif name in ['T-5', 'T-6', 'T-7', 'T-8']:
        return 'Mix'
    else:
        return 'Unknown'

df['treatment_type'] = df['name'].apply(get_treatment_type)

summary_stats = []
for (name, control, treatment_type), group in df.groupby(['name', 'control', 'treatment_type']):
    for gas in ['CO2', 'N2O', 'CH4']:
        mean_val = group[gas+"-Mean"].values[0]
        std_val = group[gas+"-STD"].values[0]

        summary_stats.append({
            'name': name,
            'control': control,
            'treatment_type': treatment_type,
            'gas': gas,
            'mean': mean_val,
            'std': std_val
        })
df_summary = pd.DataFrame(summary_stats)
df_summary['mean'] = pd.to_numeric(df_summary['mean'], errors='coerce')
df_summary['std'] = pd.to_numeric(df_summary['std'], errors='coerce')

#%%
plt.clf()
size1 = 16
fontdict = {'weight': 'bold','size':size1,'family':'SimHei'}
mpl.rcParams.update(
    {
        'text.usetex': False,
        'font.sans-serif': ['Arial'],
        'mathtext.fontset': 'stix',
        'font.size': 16                    
    }
)

fig = plt.figure(figsize=(15,8))

gs = plt.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.1, wspace=0.30)

gases = ['CO2', 'N2O', 'CH4']
treatment_types = ['Single', 'Mix']
colors = ['#5BB3E4', '#009D73', '#E59F24']
gas_labels = ['CO$_2$', 'N$_2$O', 'CH$_4$']
markers = ['o', 's', '^', 'D']

for i, treatment_type in enumerate(treatment_types):
    for j, gas in enumerate(gases):
        ax =fig.add_subplot(gs[i, j])
        
        if treatment_type == 'Single':
            names = ['T-1', 'T-2', 'T-3', 'T-4']

        else:  # Mix
            names = ['T-5', 'T-6', 'T-7', 'T-8']

        for idx, name in enumerate(names):
            data_subset = df_summary[(df_summary['treatment_type'] == treatment_type) & 
                                     (df_summary['gas'] == gas) &
                                     (df_summary['name'] == name)].copy()
            
            if len(data_subset) > 0:
                data_subset = data_subset.sort_values('control')
                x_labels = ['CK','OTC']
                x_pos = [0, 1]
                
                means = data_subset['mean'].values
                stds = data_subset['std'].values

                if i == 0:  # single
                    ax.errorbar(x_pos, means, yerr=stds, 
                              label=name, 
                              color=colors[j],
                              marker=markers[idx],
                              markersize=8,
                              markerfacecolor=colors[j], 
                              markeredgecolor=colors[j], 
                              capsize=5,
                              capthick=2,
                              linewidth=2)
                else: 
                    ax.errorbar(x_pos, means, yerr=stds, 
                              label=name, 
                              color=colors[j],
                              marker=markers[idx],
                              markersize=8,
                              markerfacecolor='white', 
                              markeredgecolor=colors[j],
                              markeredgewidth=2,
                              capsize=5,
                              capthick=2,
                              linewidth=2)
                
                ax.set_xlim(-0.3, 1.3)
                ax.set_xticks([0, 1])
                if i == 1:  # micro
                    ax.set_xticklabels(x_labels, fontsize=18)
                    
                else:
                    ax.set_xticklabels([])
                    ax.set_title(gas_labels[j], fontsize=18, fontweight='normal')
                if (i == 0) & (j==0) :
                    ax.text(-0.4, 0.5, 'Single aggregate system', transform=ax.transAxes,
                            fontsize=18, fontweight='normal', rotation=90,
                            va='center', ha='center')
                if (i == 1) & (j==0) :
                    ax.text(-0.4, 0.5, 'Mix aggregate system', transform=ax.transAxes,
                            fontsize=18, fontweight='normal', rotation=90,
                            va='center', ha='center')
                ax.grid(True, alpha=0.3, linestyle='--')
                if j == 0:
                    ax.set_ylim(8, 16)
                elif j == 1:
                    ax.set_ylim(15, 55)
                elif j == 2:
                    ax.set_ylim(-5, 65)
        ylabels = ['CO$_2$ Flux (mg C kg soil d$^{-1}$)', 'N$_2$O Flux (μg N kg soil d$^{-1}$)', 'CH$_4$ Flux (μg C kg soil d$^{-1}$)']

        if j == 0:
            ax.set_ylabel(ylabels[0], fontsize=14)
        elif j == 1:
            ax.set_ylabel(ylabels[1], fontsize=14)
        elif j == 2:
            ax.set_ylabel(ylabels[2], fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        if i == 0 and j == 2:
            custom_handles = []
            for idx, name in enumerate(names):
                if treatment_type == 'Single':
                    marker_style = markers[idx]
                else:
                    marker_style = markers[idx]
                handle = plt.Line2D([0], [0], 
                                marker=marker_style, 
                                color='black',
                                linestyle='', 
                                markersize=12, 
                                label=name)
                custom_handles.append(handle)

            ax.legend(handles=custom_handles,
                    fontsize=18,
                    loc='upper left',
                    title='Treatments',          
                    frameon=False,               
                    bbox_to_anchor=(1.05, 1.0), 
                    title_fontsize=18)
            
        if i == 1 and j == 2:
            custom_handles = []
            for idx, name in enumerate(names):
                if treatment_type == 'Mix':
                    marker_style = markers[idx]
                else:
                    marker_style = markers[idx]
                
                handle = plt.Line2D([0], [0], 
                                marker=marker_style, 
                                color='black',
                                markerfacecolor='white',
                                markeredgecolor='black',
                                markeredgewidth=2,
                                linestyle='',
                                markersize=12,
                                label=name)
                custom_handles.append(handle)
    
            ax.legend(handles=custom_handles,
                    fontsize=18,
                    loc='upper left',
                    title='Treatments',          
                    frameon=False,               
                    bbox_to_anchor=(1.05, 1.0),
                    title_fontsize=18)
fig.suptitle("A: Effect of long-term warming on the GHG emissions", 
             fontsize=16, fontweight='bold', x= 0.32, y=0.95)

def calculate_percentage_change(treatment_names, gas_names, df_summary, n=3):
    heatmap_data = []
    p_values = []

    for gas in gas_names:
        row_data = []
        row_p_values = []

        for name in treatment_names:
            ck_row = df_summary[
                (df_summary['name'] == name)
                & (df_summary['control'] == 'CK')
                & (df_summary['gas'] == gas)
            ]
            otc_row = df_summary[
                (df_summary['name'] == name)
                & (df_summary['control'] == 'OTC')
                & (df_summary['gas'] == gas)
            ]

            if not ck_row.empty and not otc_row.empty:
                ck_mean = ck_row['mean'].values[0]
                ck_std = ck_row['std'].values[0]
                otc_mean = otc_row['mean'].values[0]
                otc_std = otc_row['std'].values[0]

                if ck_mean != 0 and not np.isnan(ck_mean):
                    percentage_change = (
                        (otc_mean - ck_mean) / ck_mean
                    ) * 100
                else:
                    percentage_change = np.nan

                # 2. Welch's t-test
                if not (np.isnan(ck_std) or np.isnan(otc_std)):
                    _, p_value = stats.ttest_ind_from_stats(
                        mean1=otc_mean,
                        std1=otc_std,
                        nobs1=n,
                        mean2=ck_mean,
                        std2=ck_std,
                        nobs2=n,
                        equal_var=False,
                    )
                else:
                    p_value = np.nan

                row_data.append(percentage_change)
                row_p_values.append(p_value)
            else:
                row_data.append(np.nan)
                row_p_values.append(np.nan)

        heatmap_data.append(row_data)
        p_values.append(row_p_values)

    return np.array(heatmap_data), np.array(p_values)

single_names = ['T-1', 'T-2', 'T-3', 'T-4']
mix_names = ['T-5', 'T-6', 'T-7', 'T-8']

gas_names_filtered = ['CO2', 'N2O', 'CH4'] 
gas_labels_filtered = ['CO$_2$', 'N$_2$O', 'CH$_4$']

single_heatmap, single_pvalues = calculate_percentage_change(single_names, gas_names_filtered,df_summary)
mix_heatmap, mix_pvalues = calculate_percentage_change(mix_names, gas_names_filtered,df_summary)

pos = gs[1, 0].get_position(fig) 
ax_single_heatmap = fig.add_axes([pos.x0,pos.y0-0.75,pos.width*1.7,pos.height*2.])
ax_mix_heatmap = fig.add_axes([pos.x0+0.43,pos.y0-0.75,pos.width*1.7,pos.height*2.])

heatmap_xlabels = ['T-1', 'T-2', 'T-3', 'T-4']
mix_xlabels = ['T-5', 'T-6', 'T-7', 'T-8']

single_heatmap_T = single_heatmap
mix_heatmap_T = mix_heatmap

single_heatmap_df = pd.DataFrame(single_heatmap_T, 
                                 index=gas_labels_filtered, 
                                 columns=heatmap_xlabels)
mix_heatmap_df = pd.DataFrame(mix_heatmap_T, 
                              index=gas_labels_filtered, 
                              columns=mix_xlabels)

def create_annotation_matrix(heatmap_data, pvalues_data):
    annotation_matrix = []
    for i in range(heatmap_data.shape[0]):
        row = []
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]
            p_value = pvalues_data[i, j]
            
            if np.isnan(value):
                row.append('')
            else:
                if p_value < 0.001:
                    sig = '***'
                elif p_value < 0.01:
                    sig = '**'
                elif p_value < 0.05:
                    sig = '*'
                else:
                    sig = 'n.s.'

                if abs(value) < 0.1: 
                    text = f'{value:.2f}%\n{sig}'
                else:
                    text = f'{value:.2f}%\n{sig}'
                row.append(text)
        annotation_matrix.append(row)
    return annotation_matrix

single_annot = create_annotation_matrix(single_heatmap_T, single_pvalues)
mix_annot = create_annotation_matrix(mix_heatmap_T, mix_pvalues)

sns.heatmap(single_heatmap_df, 
            ax=ax_single_heatmap,
            cmap='RdBu_r',
            center=0,
            vmin=-60, vmax=60,
            annot=single_annot,
            fmt='',
            annot_kws={'fontsize': 16},
            cbar=False,
            linewidths=1,
            linecolor='white',
            square=True)
for spine in ax_single_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color('black')

ax_single_heatmap.set_title("B: Response ratio of long-term warming of GHG emissions",
                          fontsize=16, fontweight='bold', pad=12, x=0.62)

im = sns.heatmap(mix_heatmap_df, 
            ax=ax_mix_heatmap,
            cmap='RdBu_r',
            center=0,
            vmin=-60, vmax=60,
            annot=mix_annot,
            fmt='',
            annot_kws={'fontsize': 16},
            cbar=False,
            linewidths=1,
            linecolor='white', 
            square=True)
for spine in ax_mix_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color('black')


cm = im.collections[0]
pos2 = ax_mix_heatmap.get_position() 
cbar_pos = fig.add_axes([pos2.x1+0.02,pos2.y0,pos2.width*0.04,pos2.height])
cbar = fig.colorbar(cm, cax=cbar_pos, orientation='vertical',extend='both')

ax_single_heatmap.tick_params(axis='both', labelsize=18)
ax_mix_heatmap.tick_params(axis='both', labelsize=18)
ax_mix_heatmap.set_yticklabels([])

plt.tight_layout()
plt.savefig('./fig5.png', dpi=300, bbox_inches='tight', facecolor='white')

# %%
