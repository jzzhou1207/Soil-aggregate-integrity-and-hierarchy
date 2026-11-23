# %%
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.patches as mpatches
# %%
data_str = """Treatment\tMacroIntact\tmicroIntact\tcontrol\tIncubation temperature\tCO2-Mean\tN2O-Mean\tCH4-Mean\tCO2-STD\tN2O-STD\tCH4-STD
T-1\tIntact\tN.S.\tCK\t9.4℃\t10.59333\t37.55\t39.63\t0.2275\t1.485\t1.79
T-2\tCrushed\tN.S.\tCK\t9.4℃\t10.67167\t26.8\t33.20167\t0.35501\t1.235\t1.39
T-3\tN.S.\tIntact\tCK\t9.4℃\t15.34\t35.485\t34.71\t0.285\t1.245\t1.275
T-4\tN.S.\tCrushed\tCK\t9.4℃\t13.39\t34.125\t36.265\t0.33\t0.8\t1.7
T-5\tIntact\tIntact\tCK\t9.4℃\t12.72\t46.35333\t47.88333\t0.3\t1.3575\t1.8025
T-6\tIntact\tCrushed\tCK\t9.4℃\t10.61833\t26.225\t0\t0.3425\t1.195\t0
T-7\tCrushed\tIntact\tCK\t9.4℃\t9.32833\t17.59\t19.295\t0.2175\t0.605\t1.015
T-8\tCrushed\tCrushed\tCK\t9.4℃\t9.94667\t29.705\t0\t0.16251\t0.58\t0
T-1\tIntact\tN.S.\tCK\t19.4℃\t16.43\t35.925\t0\t0.31\t1.34\t0
T-2\tCrushed\tN.S.\tCK\t19.4℃\t15.38167\t30.37167\t0\t0.3575\t0.8225\t0
T-3\tN.S.\tIntact\tCK\t19.4℃\t18.025\t27.195\t0\t0.45\t1.425\t0
T-4\tN.S.\tCrushed\tCK\t19.4℃\t14.235\t40.94\t24.655\t0.315\t1.68\t0.485
T-5\tIntact\tIntact\tCK\t19.4℃\t14.875\t40.73\t29.96\t0.39\t1.495\t0.8
T-6\tIntact\tCrushed\tCK\t19.4℃\t23.81\t80.685\t59.125\t0.445\t2.06\t1.025
T-7\tCrushed\tIntact\tCK\t19.4℃\t18.26\t52.135\t0\t0.445\t1.94\t0
T-8\tCrushed\tCrushed\tCK\t19.4℃\t14.335\t34.005\t0\t0.355\t0.99\t0
T-1\tIntact\tN.S.\tOTC\t19.4℃\t17.385\t51.645\t26.39\t0.47\t1.275\t0.75
T-2\tCrushed\tN.S.\tOTC\t19.4℃\t16.99\t49.215\t0\t0.515\t1.565\t0
T-3\tN.S.\tIntact\tOTC\t19.4℃\t13.72\t26.275\t24.55\t0.225\t0.84\t0.505
T-4\tN.S.\tCrushed\tOTC\t19.4℃\t14.275\t39.23\t30.14\t0.44\t1.985\t0.665
T-5\tIntact\tIntact\tOTC\t19.4℃\t15.81\t58.06\t57.095\t0.52\t0.915\t0.95
T-6\tIntact\tCrushed\tOTC\t19.4℃\t21.06\t36.555\t36.39\t0.575\t0.52\t0.925
T-7\tCrushed\tIntact\tOTC\t19.4℃\t18.235\t45.29\t0\t0.64\t2.035\t0
T-8\tCrushed\tCrushed\tOTC\t19.4℃\t18.75\t46.245\t96.87\t0.43\t1.835\t2.935"""

df = pd.read_csv(io.StringIO(data_str), sep='\t')

# %%

def get_sig_label(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'n.s.'

def calc_ind_ttest(m1, s1, m2, s2, n1=3, n2=3):
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    if se == 0:
        return 1.0
    t_stat = (m1 - m2) / se
    df_deg = n1 + n2 - 2
    return 2 * (1 - stats.t.cdf(abs(t_stat), df=df_deg))

gases = ['CO2', 'N2O', 'CH4']
ylabels = ['CO$_2$ Flux (mg C kg soil d$^{-1}$)', 'N$_2$O Flux (μg N kg soil d$^{-1}$)', 'CH$_4$ Flux (μg C kg soil d$^{-1}$)']
colors = ['#4C72B0', '#55A868', "#DABA81", '#C44E52'] 

row_configs = [
    {
        'name': 'Single Aggregates',
        'treatments': ['T-1', 'T-2', 'T-3', 'T-4'],
        'hatch': '',
    },
    {
        'name': 'Mix Aggregates',
        'treatments': ['T-5', 'T-6', 'T-7', 'T-8'],
        'hatch': '//',
    },
]
ymax_dict_abs = {
    g: (df[f'{g}-Mean'] + df[f'{g}-STD']).max() for g in gases
}
#%%
fig = plt.figure(figsize=(16, 10))
outer_gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

for r_idx, r_cfg in enumerate(row_configs):
    t_list = r_cfg['treatments']
    hatch_style = r_cfg['hatch']
    x_pos = np.arange(len(t_list))

    for c_idx, gas in enumerate(gases):
        inner_gs = outer_gs[r_idx, c_idx].subgridspec(
            2, 2, width_ratios=[1, 1], hspace=0.05, wspace=0.6
        )

        ax_left = fig.add_subplot(inner_gs[:, 0])
        ax_top_right = fig.add_subplot(inner_gs[0, 1])
        ax_bot_right = fig.add_subplot(inner_gs[1, 1])

        # 9.4℃ CK
        df_94 = df[
            (df['Treatment'].isin(t_list))
            & (df['control'] == 'CK')
            & (df['Incubation temperature'] == '9.4℃')
        ].sort_values('Treatment')
        val_94 = df_94[f'{gas}-Mean'].values
        std_94 = df_94[f'{gas}-STD'].values

        # 19.4℃ CK
        df_194_ck = df[
            (df['Treatment'].isin(t_list))
            & (df['control'] == 'CK')
            & (df['Incubation temperature'] == '19.4℃')
        ].sort_values('Treatment')
        val_194_ck = df_194_ck[f'{gas}-Mean'].values
        std_194_ck = df_194_ck[f'{gas}-STD'].values

        # 19.4℃ OTC
        df_194_otc = df[
            (df['Treatment'].isin(t_list))
            & (df['control'] == 'OTC')
            & (df['Incubation temperature'] == '19.4℃')
        ].sort_values('Treatment')
        val_194_otc = df_194_otc[f'{gas}-Mean'].values
        std_194_otc = df_194_otc[f'{gas}-STD'].values

        pct_ck = np.where(val_94 > 0, ((val_194_ck - val_94) / val_94) * 100, 0)
        pct_otc = np.where(
            val_94 > 0, ((val_194_otc - val_94) / val_94) * 100, 0
        )

        ax_left.bar(
            x_pos,
            val_94,
            yerr=std_94,
            color=colors,
            capsize=3,
            edgecolor='k',
            alpha=0.85,
            width=0.98,
            hatch=hatch_style,
        )
        ax_left.set_xticks([x_pos.mean()])
        ax_left.set_xticklabels(
            ['Ambient'], fontsize=12, fontweight='normal'
        )
        ax_left.set_ylim(0, ymax_dict_abs[gas])
        ax_left.set_ylabel(
            f"{ylabels[c_idx]}", fontsize=12, fontweight='normal'
        )
        ax_left.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax_left.set_title('CK', fontsize=13, fontweight='normal',x=0.8,y=0.92)

        bars_ck = ax_top_right.bar(
            x_pos, pct_ck, color=colors, edgecolor='k', alpha=0.85,hatch=hatch_style,width=0.98,
        )
        ax_top_right.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax_top_right.set_xticks([])
        ax_top_right.set_title('CK', fontsize=12, fontweight='normal',x=0.8,y=0.85)
        ax_top_right.set_ylabel(
            r'Response ratio (%)', fontsize=12, fontweight='normal'
        )
        ax_top_right.grid(True, linestyle='--', alpha=0.3, axis='y')

        y_max_ck, y_min_ck = (
            max(pct_ck.max(), 10),
            min(pct_ck.min(), -10),
        )
        for i in range(len(t_list)):
            p_val = calc_ind_ttest(
                val_194_ck[i], std_194_ck[i], val_94[i], std_94[i]
            )
            sig = get_sig_label(p_val)

            y_val = pct_ck[i]
            if y_val >= 0:
                ax_top_right.text(
                    x_pos[i],
                    y_val + (y_max_ck - y_min_ck) * 0.05,
                    sig,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold',
                )
            else:
                ax_top_right.text(
                    x_pos[i],
                    y_val - (y_max_ck - y_min_ck) * 0.05,
                    sig,
                    ha='center',
                    va='top',
                    fontsize=8,
                    fontweight='bold',
                )

        bars_otc = ax_bot_right.bar(
            x_pos, pct_otc, color=colors, edgecolor='k', alpha=0.85,hatch=hatch_style,width=0.98,
        )
        ax_bot_right.set_title('OTC', fontsize=13, fontweight='normal',x=0.8,y=0.85)
        ax_bot_right.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax_bot_right.set_xticks([x_pos.mean()])
        ax_bot_right.set_xticklabels(
            ['+10℃'], fontsize=12, fontweight='normal'
        )
        ax_bot_right.set_ylabel(
            r'Response ratio (%)', fontsize=12, fontweight='normal'
        )
        ax_bot_right.grid(True, linestyle='--', alpha=0.3, axis='y')

        y_max_otc, y_min_otc = max(pct_otc.max(), 10), min(pct_otc.min(), -10)
        for i in range(len(t_list)):
            p_val = calc_ind_ttest(
                val_194_otc[i], std_194_otc[i], val_94[i], std_94[i]
            )
            sig = get_sig_label(p_val)

            y_val = pct_otc[i]
            if y_val >= 0:
                ax_bot_right.text(
                    x_pos[i],
                    y_val + (y_max_otc - y_min_otc) * 0.05,
                    sig,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold',
                )
            else:
                ax_bot_right.text(
                    x_pos[i],
                    y_val - (y_max_otc - y_min_otc) * 0.05,
                    sig,
                    ha='center',
                    va='top',
                    fontsize=8,
                    fontweight='bold',
                )

        ax_bot_right.set_ylim(
            min(y_min_otc,y_min_ck) - abs(y_min_otc) * 0.35, max(y_max_otc,y_max_ck) + abs(y_max_otc) * 0.35
        )
        ax_top_right.set_ylim(
            min(y_min_otc,y_min_ck) - abs(y_min_otc) * 0.35, max(y_max_otc,y_max_ck) + abs(y_max_otc) * 0.35
        )
        if gas == 'CO2' and  (r_cfg['name'] == 'Single Aggregates'):
            ax_top_right.set_ylim(-25, 80)
            ax_bot_right.set_ylim(-25, 80)
        if gas == 'N2O' and  (r_cfg['name'] == 'Mix Aggregates'):
            ax_top_right.set_ylim(-50, 250)
            ax_bot_right.set_ylim(-50, 250)
        if gas == 'CH4' and  (r_cfg['name'] == 'Single Aggregates'):
            ax_top_right.set_ylim(-125, 25)
            ax_bot_right.set_ylim(-125, 25)
        if gas == 'CH4' and  (r_cfg['name'] == 'Mix Aggregates'):
            ax_top_right.set_ylim(-150, 50)
            ax_bot_right.set_ylim(-150, 50)

legend_patches = [
    # Single Aggregates
    mpatches.Patch(
        facecolor=colors[0],
        edgecolor='k',
        label='T-1',
        alpha=0.85,
    ),
    mpatches.Patch(
        facecolor=colors[1],
        edgecolor='k',
        label='T-2',
        alpha=0.85,
    ),
    mpatches.Patch(
        facecolor=colors[2],
        edgecolor='k',
        label='T-3',
        alpha=0.85,
    ),
    mpatches.Patch(
        facecolor=colors[3],
        edgecolor='k',
        label='T-4',
        alpha=0.85,
    ),
    # Mix Aggregates
    mpatches.Patch(
        facecolor=colors[0],
        edgecolor='k',
        hatch='//',
        label='T-5',
        alpha=0.85,
    ),
    mpatches.Patch(
        facecolor=colors[1],
        edgecolor='k',
        hatch='//',
        label='T-6',
        alpha=0.85,
    ),
    mpatches.Patch(
        facecolor=colors[2],
        edgecolor='k',
        hatch='//',
        label='T-7',
        alpha=0.85,
    ),
    mpatches.Patch(
        facecolor=colors[3],
        edgecolor='k',
        hatch='//',
        label='T-8',
        alpha=0.85,
    ),
]

fig.legend(
    handles=legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.78, 0.51),
    ncol=4,
    fontsize=12,
    frameon=False,
)
fig.suptitle(
    'A: Effect of incurbation warming on the GHG emissions in single aggregates',
    fontsize=14,
    fontweight='bold',
    y=0.92,
    x=0.35
)

fig.text(0.34, 0.47, 
         'B: Effect of incurbation warming on the GHG emissions in mix aggregates',
          ha='center', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('./fig6.png', dpi=300, bbox_inches='tight')

# %%
