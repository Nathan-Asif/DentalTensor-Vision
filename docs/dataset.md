# Dataset Provenance & Curation Strategy

## 1. Datasets Utilized

### Primary Pathology Dataset (`oral-disease.yolov11`)
- **Total Images**: 10,698 images
- **Splits**:
  - Train: 8,558 images
  - Validation: 1,070 images
  - Test: 1,070 images
- **Total Annotations**: 62,720 bounding boxes across oral disease categories
- **Control Images**: 570 empty-label true negatives (5.33% of the dataset)
- **Source**: Roboflow Universe (`di-qidb9/oral-disease-tabrb`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

### Auxiliary Healthy Dentition Pools
- **Sources**: `Dental Data Set.yolov11` and `Penyakit Gigi Skripsi.yolov11` (Roboflow Universe)
- **Role**: Source pools for hard-negative mining of clean teeth and gums.
- **License**: CC BY 4.0

---

## 2. Hard-Negative Mining & Curation Strategy

Rather than naively concatenating entire external datasets—which frequently contain noisy, inconsistent, or mislabeled boxes—an audited mining strategy was enforced:

1. **False-Positive Profiling**: The baseline model was run on hundreds of verified healthy oral images. Images triggering false-positive alerts on clean dentition were identified.
2. **Manual Review & Approval**: Only clean images with zero pathology were selected as approved true negative control samples.
3. **Balanced Injection**: 27 unique, manually verified negative controls were added with balanced repetition into the training set (expanding training size to 8,666 images).
4. **Leakage Protection**: The benchmark validation split (1,070 images) and test split (1,070 images) remained 100% UNTOUCHED and isolated throughout all training phases.
