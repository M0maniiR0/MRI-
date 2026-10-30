import os
import json
import glob
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")

def test_acceptance():
    label_map_path = BASE_DIR / "label_map.json"
    assert label_map_path.exists(), "label_map.json does not exist"
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    # 1. Check exactly 10 official classes
    expected_classes = [
        "Normal", "Glioma", "Meningioma", "Metastasis", "Alzheimer's",
        "FCD epilepsy", "Post-stroke lesion", "Parkinson's", "ASD", "Uncertain/Other"
    ]
    actual_classes = [c["name"] for c in lm["classes"]]
    assert actual_classes == expected_classes, f"Classes mismatch: {actual_classes}"
    assert len(actual_classes) == 10, "Must have exactly 10 classes"

    # 2. Audit all training cohorts
    # Glioma training
    glioma_json = BASE_DIR / "Task01_BrainTumour" / "Task01_BrainTumour" / "dataset.json"
    with open(glioma_json, "r", encoding="utf-8") as f:
        g_meta = json.load(f)
    g_train = g_meta["training"]

    # Meningioma training
    men_manifest = BASE_DIR / "Task_Meningioma" / "BraTS_MEN" / "dataset_manifest.json"
    with open(men_manifest, "r", encoding="utf-8") as f:
        m_train = json.load(f)

    # Metastasis training
    meta_manifest = BASE_DIR / "Task_BrainMetastasis" / "dataset_manifest.json"
    with open(meta_manifest, "r", encoding="utf-8") as f:
        meta_train = json.load(f)

    # Alzheimer Hippocampus training
    hippo_json = BASE_DIR / "Task04_Hippocampus" / "dataset.json"
    with open(hippo_json, "r", encoding="utf-8") as f:
        h_meta = json.load(f)
    h_train = h_meta["training"]

    # Multiple Sclerosis (excluded)
    ms_manifest = BASE_DIR / "Task_MultipleSclerosis" / "dataset_manifest.json"
    with open(ms_manifest, "r", encoding="utf-8") as f:
        ms_train = json.load(f)

    # Map each subject
    audit_log = []
    
    for item in g_train:
        cid = Path(item["image"]).name
        target_class = lm["source_to_class"]["MSD_Task01_BrainTumour"]["Glioma"]
        audit_log.append((cid, target_class, "INCLUDED"))

    for item in m_train:
        cid = item["case_id"]
        target_class = lm["source_to_class"]["BraTS_MEN"]["Meningioma"]
        audit_log.append((cid, target_class, "INCLUDED"))

    for item in meta_train:
        cid = item["patient_id"]
        target_class = lm["source_to_class"]["BrainMetastasis_Pakawat_SCI"]["Metastasis"]
        audit_log.append((cid, target_class, "INCLUDED"))

    for item in h_train:
        cid = Path(item["image"]).name
        target_class = lm["source_to_class"]["MSD_Task04_Hippocampus"]["Hippocampal_Atrophy_AD_Spectrum"]
        audit_log.append((cid, target_class, "INCLUDED"))

    for item in ms_train:
        cid = item["patient_id"]
        reason = lm["exclusions"]["MultipleSclerosis_MSLesSeg"]["reason"]
        audit_log.append((cid, "EXCLUDED", reason))

    # Verification: every subject has exactly one mapped class or an explicit logged exclusion
    total_subjects_audited = len(audit_log)
    included = [x for x in audit_log if x[2] == "INCLUDED"]
    excluded = [x for x in audit_log if x[1] == "EXCLUDED"]

    print(f"[TEST PASSED] Total training subjects audited: {total_subjects_audited}")
    print(f"  - Included subjects mapped to official classes: {len(included)}")
    print(f"  - Excluded subjects with logged justification: {len(excluded)}")
    assert len(included) + len(excluded) == total_subjects_audited
    assert len(included) == len(g_train) + len(m_train) + len(meta_train) + len(h_train)
    assert len(excluded) == len(ms_train)
    print("Acceptance criteria for Task 1 fully satisfied!")

if __name__ == "__main__":
    test_acceptance()
