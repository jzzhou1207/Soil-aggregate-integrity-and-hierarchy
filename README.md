# Soil aggregate integrity and hierarchy offer a potential pathway for managing greenhouse gas emissions under warming

## Overview

This repository accompanies the manuscript **Soil aggregate integrity and hierarchy offer a potential pathway for managing greenhouse gas emissions under warming**, prepared for submission to *Carbon Balance and Management*. It provides the summary datasets and Python plotting scripts for Figures 3–6 and documents the intended data processing and figure-reproduction workflow.

The study examines how aggregate size, structural integrity, and the arrangement of macroaggregates and microaggregates influence soil carbon dioxide (CO₂), nitrous oxide (N₂O), methane (CH₄) emissions, and their combined global warming potential (GWP) under different warming conditions. It combines soils from an approximately ten-year open-top-chamber (OTC) field-warming experiment in a Qinghai–Tibetan Plateau alpine meadow with controlled laboratory aggregate reconstitution and incubation.

The management motivation is to evaluate whether maintaining or modifying soil aggregate structure may influence greenhouse gas responses to warming. 

> **Repository status.** The four `figure_*.csv` tables included here were reconstructed directly from numeric values embedded in the four supplied plotting scripts.

## Repository structure

```text
.
├── README.md                  # Project description and data/code dictionary
├── requirements.txt           # Python dependencies
├── data/
│   ├── figure_3.csv           # Figure 3: single-fraction aggregate disruption
│   ├── figure_4.csv           # Figure 4: mixed-aggregate observed vs. additive predictions
│   ├── figure_5.csv           # Figure 5: long-term field-warming comparison
│   ├── figure_6.csv           # Figure 6: laboratory and combined warming responses
│   └── gwp.csv                # Expected GWP dataset
├── code/
│   ├── fig3.py                # Generate Figure 3
│   ├── fig4.py                # Generate Figure 4
│   ├── fig5.py                # Generate Figure 5
│   └── fig6.py                # Generate Figure 6
```

## Experimental design and treatment definitions

**Field warming:** `CK` denotes the ambient field control; `OTC` denotes soils collected from open-top chamber plots exposed to long-term field warming. **Laboratory incubation:** 9.4 °C is the ambient incubation treatment; 19.4 °C represents a +10 °C laboratory warming treatment.

Eight aggregate configurations are examined:

| Treatment | Aggregate system | Macroaggregates | Microaggregates |
|:--|:--|:--|:--|
| T-1 | Single fraction | Intact | Not included |
| T-2 | Single fraction | Crushed | Not included |
| T-3 | Single fraction | Not included | Intact |
| T-4 | Single fraction | Not included | Crushed |
| T-5 | Mixed fractions | Intact | Intact |
| T-6 | Mixed fractions | Intact | Crushed |
| T-7 | Mixed fractions | Crushed | Intact |
| T-8 | Mixed fractions | Crushed | Crushed |

## Software requirements and quick start

Python 3.10+ is recommended. The scripts import `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, and `statsmodels`. Fonts and visual appearance may vary across operating systems.

```bash
# Run from the repository root
pip install -r requirements.txt

python code/fig3.py
python code/fig4.py
python code/fig5.py
python code/fig6.py
```
