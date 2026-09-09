# 📋 Dataset Description: 3D CAS Images (200 Samples)
**Generated in Phase 3**  
**Source Location**: `H:\3D CT Images for Coronary Artery Segmentation (200 Samples)`  

---

| Field | Value |
| :--- | :--- |
| **Dataset Name** | **3D CAS Images (200 Samples)** |
| **Number of Cases** | **200** |
| **File Format** | NIfTI-1 (`.nii` uncompressed) |
| **Imaging Modality** | 3D Cardiac Computed Tomography Angiography (CCTA) |
| **Source / Curator** | ImageCAS Repository (Format A subset, cases 1–200) |
| **License** | Creative Commons BY-NC-SA 4.0 |
| **Source Link** | [ImageCAS GitHub Repository](https://github.com/XiaoweiXu/ImageCAS-A-Large-Coronary-Artery-Dataset) |
| **Dataset Description** | Contrast-enhanced 3D CCTA volumetric scans depicting complete coronary arterial trees |
| **Annotation Type** | Binary voxel ground-truth segmentations (1: Coronary Tree, 0: Background) |
| **Anatomical Targets** | Left Main (LM), Left Anterior Descending (LAD), Left Circumflex (LCx), Right Coronary Artery (RCA) |
| **Required Preprocessing** | Standardized to RAS orientation, 0.5 mm isotropic spacing, HU intensity window [-100, 800] |
| **Primary Dataset Overlap** | **Verified repackaging of ImageCAS Cases 1–200**. Overlaps with 115 primary train cases and 19 val cases; 66 cases are clean/unseen. |
| **Reporting Standard** | Explicit two-tier reporting: Full Cohort ($N=200$) alongside Leakage-Free Unseen Cohort ($N=66$). |

---

### Overlap & Provenance Audit Statement (CLAIM 2024 / Q1 Compliance)
> *"The 200-case 3D CAS collection was formally audited against our primary 950-case benchmark partition. Provenance tracing confirms this collection represents Cases 1 through 200 of the ImageCAS database. Within this cohort, 115 cases were included in the training partition of the 200-epoch benchmark models, 19 cases in validation, and 66 cases represent cases excluded during initial quality-filtering that were never exposed to model optimization. Consequently, to maintain absolute scientific rigor and avoid data leakage artifacts, all evaluation metrics are reported both for the complete 200-case cohort and disaggregated into the genuine unseen sub-cohort ($N=66$) and training-recall sub-cohort ($N=115$)."*
