# OmicsInsight — Analysis Report

**Generated:** 2026-04-25T16:49:47.903840

## Dataset Overview

- **Samples:** 27
- **Genes (raw):** 46102
- **Features (after preprocessing):** 500
- **Metadata columns:** title, geo_accession, status, submission_date, last_update_date, type, channel_count, source_name_ch1, organism_ch1, treatment_protocol_ch1, growth_protocol_ch1, molecule_ch1, extract_protocol_ch1, taxid_ch1, description, data_processing, platform_id, contact_name, contact_email, contact_department, contact_institute, contact_address, contact_city, contact_state, contact_zip/postal_code, contact_country, data_row_count, instrument_model, library_selection, library_source, library_strategy, relation, supplementary_file_1, treatment, time_point_minutes, cultivar, replicate, sample_id
- **Count data file:** `C:\Users\ibrah\Desktop\sync_150625\OmicsInsight\dataset\GSE124666\GSE124666_NGS_000247_countData.txt`
- **Metadata file:** `C:\Users\ibrah\Desktop\sync_150625\OmicsInsight\dataset\GSE124666\GSE124666_series_matrix.txt`

## Preprocessing Summary

- **Log2 transform:** True
- **Min total count filter:** 10
- **Genes after count filter:** 35931
- **Min variance filter:** 0.0
- **Genes after variance filter:** 35931
- **Max features selected:** 500
- **Final feature count:** 500

## PCA Results

| Component | Explained Variance | Cumulative |
|-----------|-------------------|------------|
| PC1 | 0.4123 | 0.4123 |
| PC2 | 0.2652 | 0.6775 |
| PC3 | 0.0922 | 0.7697 |
| PC4 | 0.0395 | 0.8092 |
| PC5 | 0.0252 | 0.8344 |
| PC6 | 0.0182 | 0.8526 |
| PC7 | 0.0136 | 0.8662 |
| PC8 | 0.0133 | 0.8795 |
| PC9 | 0.0123 | 0.8917 |
| PC10 | 0.0102 | 0.9020 |

## Clustering Results

### KMeans

- **Silhouette score:** 0.2655
- **Adjusted Rand Index (vs true labels):** 0.2614
- **Cluster sizes:** {0: 6, 1: 12, 2: 9}

### Agglomerative

- **Silhouette score:** 0.254
- **Adjusted Rand Index (vs true labels):** 0.192
- **Cluster sizes:** {0: 10, 1: 11, 2: 6}

## Classification Results (Exploratory)

> **Note:** With only 27 samples and Leave-One-Out cross-validation, these results are exploratory and should not be interpreted as generalizable predictive performance.

### LogisticRegression

- **Accuracy (LOO-CV):** 0.7778
- **Macro F1 (LOO-CV):** 0.7778
- **Classes:** ['Cyprosulfamide', 'DMSO', 'Metcamifen']

**Confusion Matrix:**

| | Pred: Cyprosulfamide | Pred: DMSO | Pred: Metcamifen |
|---|---|---|---|
| **Cyprosulfamide** | 6 | 3 | 0 |
| **DMSO** | 3 | 6 | 0 |
| **Metcamifen** | 0 | 0 | 9 |

### RandomForest

- **Accuracy (LOO-CV):** 0.7037
- **Macro F1 (LOO-CV):** 0.7037
- **Classes:** ['Cyprosulfamide', 'DMSO', 'Metcamifen']

**Confusion Matrix:**

| | Pred: Cyprosulfamide | Pred: DMSO | Pred: Metcamifen |
|---|---|---|---|
| **Cyprosulfamide** | 5 | 3 | 1 |
| **DMSO** | 3 | 6 | 0 |
| **Metcamifen** | 1 | 0 | 8 |

## Top Ranked Features

Top 20 features by average rank across methods:

| Rank | Feature | Avg Rank | Variance |
|------|---------|----------|----------|
| 1 | `LOC_Os07g13770` | 23.333333 | 3.347458 |
| 2 | `LOC_Os10g08540` | 30.333333 | 4.422212 |
| 3 | `LOC_Os09g34260` | 30.666667 | 4.227281 |
| 4 | `LOC_Os09g20220` | 35.0 | 12.478846 |
| 5 | `LOC_Os01g08440` | 41.666667 | 11.255801 |
| 6 | `LOC_Os01g64910` | 42.0 | 4.484396 |
| 7 | `LOC_Os07g44550` | 44.666667 | 2.326653 |
| 8 | `LOC_Os03g57200` | 45.333333 | 7.503958 |
| 9 | `LOC_Os02g16280` | 51.0 | 3.584528 |
| 10 | `LOC_Os04g46600` | 57.666667 | 2.142343 |
| 11 | `LOC_Os10g22310` | 59.0 | 2.882362 |
| 12 | `LOC_Os09g32000` | 62.0 | 10.623636 |
| 13 | `LOC_Os05g34830` | 62.0 | 4.624388 |
| 14 | `LOC_Os11g04860` | 62.0 | 2.02419 |
| 15 | `LOC_Os02g28220` | 64.0 | 2.615583 |
| 16 | `LOC_Os04g10060` | 64.0 | 3.898075 |
| 17 | `LOC_Os07g49114` | 65.0 | 2.903323 |
| 18 | `LOC_Os06g11260` | 65.0 | 5.967454 |
| 19 | `LOC_Os07g23570` | 66.333333 | 8.411058 |
| 20 | `LOC_Os07g23710` | 66.666667 | 6.247049 |

## Biological Interpretation

This analysis explored the transcriptomic response of rice (*Oryza sativa*) suspension cultures to two safener compounds (metcamifen and cyprosulfamide) versus a DMSO control, sampled at three time points (30, 90, and 240 minutes).

PCA and clustering analyses reveal the major axes of transcriptomic variation across treatment groups. Features ranked as important by both supervised (classification coefficient / importance) and unsupervised (variance) methods may represent genes whose expression is most responsive to safener treatment. These candidate genes warrant further investigation through differential expression analysis, functional annotation, and pathway enrichment using dedicated statistical frameworks (e.g., DESeq2, edgeR).

The balanced experimental design (3 treatments × 3 time points × 3 replicates = 27 samples) supports exploratory pattern discovery but is insufficient for robust biomarker validation.

## Limitations and Assumptions

1. **Small sample size (n=27):** All machine-learning results are exploratory. With 9 samples per treatment class and Leave-One-Out CV, metrics may exhibit high variance and should not be over-interpreted.

2. **No formal normalization:** Log2(x+1) transformation is applied to raw counts. For publication-grade differential expression, proper normalization (TMM, DESeq2 size factors, VST) is recommended.

3. **Feature selection bias:** Top features are selected by variance across all samples *before* classification. For rigorous biomarker discovery, feature selection should be embedded within the cross-validation loop.

4. **No pathway or functional enrichment:** Gene identifiers are reported without annotation. Downstream analysis should map them to GO terms, KEGG pathways, or rice-specific databases.

5. **Read counting caveat:** The original count data allowed multi-mapping (up to 6 gene regions per read), which may inflate counts for gene families with high sequence similarity.

6. **Exploratory scope:** This project is a reusable computational analysis workflow, not a replacement for rigorous statistical genomics pipelines. Results are hypothesis-generating.
