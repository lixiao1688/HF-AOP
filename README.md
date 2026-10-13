# HF-AOP: Multi-Algorithm Ensemble Model for Drug-Induced Heart Failure Risk Prediction

Predictive model for drug-induced heart failure based on the Adverse Outcome Pathway (AOP 558) of PDE4D inhibition, using a four-tier hierarchical ensemble (MIE → KE1 → KE2 → AO) and a two-level stacking framework.

## Overview

- **Framework**: AOP-558 (phosphodiesterase 4D inhibition leading to heart failure)
- **Four tiers**: MIE (PDE4D binding), KE1 (cAMP elevation), KE2 (calcium dysregulation), AO (heart failure)
- **Fingerprints**: MACCS, PubChem, ChemBERTa, MACCS+ChemBERTa
- **Algorithms**: SVM, Random Forest, DNN, ChemProp, AttentiveFP, GraphSAVE (tier-1); XGBoost (tier-2 stacking)
- **Validation**: 5-fold CV, leave-one-out CV (tier 2), scaffold splitting, external validation on FDA DICTrank

## Repository structure

```
data/
  MIE_PDE4D.csv                          SMILES + label for MIE tier (865 compounds)
  KE1_cAMP.csv                           SMILES + label for KE1 tier (3811 compounds)
  KE2_calcium.csv                        SMILES + label for KE2 tier (968 compounds)
  AO_heart_failure.csv                   SMILES + label for AO tier (1190 compounds)
  external_validation_DICTrank_set.csv            FDA DICTrank external validation set (SMILES + label)
  external_validation_heart_failure_matched_set.csv  Heart-failure endpoint-matched external set
  external_validation_DICTrank_predictions.csv   External validation predictions (tier probabilities)
models/
  MIE_consensus_model.pkl                Tier-1 consensus model (MIE)
  KE1_consensus_model.pkl                Tier-1 consensus model (KE1)
  KE2_consensus_model.pkl                Tier-1 consensus model (KE2)
  AO_consensus_model.pkl                 Tier-1 consensus model (AO)
  meta_model_xgb.pkl                     Tier-2 stacking meta-model (XGBoost)
scripts/
  心衰外部验证.py                         External validation (DICTrank)
  第二层XGB.py                            Tier-2 stacking (XGBoost)
  第二层堆叠.py                           Tier-2 stacking pipeline
  活性悬崖.py                             Activity cliff analysis
  AD全.py                                 Applicability domain analysis
  第一层集成汇总.py                        Tier-1 consensus aggregation
  SHAP可解释性分析.py                     SHAP interpretability
  chemprop-training&test.py              Graph neural network (ChemProp)
```

## Data sources

- **MIE**: PDE4D inhibitor data from BindingDB and ChEMBL (pChEMBL > 7.0 positive, < 5.0 negative)
- **KE1/KE2**: high-throughput screening data for cAMP elevation and calcium leakage from PubChem BioAssay
- **AO**: compounds associated with heart failure from CTD, SIDER, OFFSIDES, and MetaADEDB (multi-source evidence fusion)
- **External validation**: FDA DICTrank dataset

## Reproducibility

All models were evaluated with fixed random seeds. Hyperparameters for all algorithms are provided in the manuscript Supporting Information. To reproduce:

```bash
pip install -r requirements.txt
python scripts/第二层XGB.py
python scripts/心衰外部验证.py
```

## Citation

Sha, F.; Gong, J.; Zhang, W.; Li, N.; Li, X.; Huang, X. Multi-Algorithm Ensemble Model for Risk Prediction of Drug-Induced Heart Failure Based on Adverse Outcome Pathway. *J. Chem. Inf. Model.* (submitted).

## Disclaimer

This tool is intended for research and early risk-screening assistance only. It does not constitute medical advice and does not replace expert judgment.
