"""Read-only NIfTI dataset inventory. No patient image voxels are exported."""
from pathlib import Path
from collections import Counter
import argparse, csv, json

SUFFIXES = ('.nii', '.nii.gz')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Folder containing NIfTI files')
    ap.add_argument('--out', default='dataset_audit.json', help='JSON audit destination')
    ap.add_argument('--sample', type=int, default=20, help='Maximum file names in preview')
    args = ap.parse_args()
    root = Path(args.data).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'Not a directory: {root}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name.lower().endswith(SUFFIXES))
    ext_counts = Counter('.nii.gz' if p.name.lower().endswith('.nii.gz') else '.nii' for p in files)
    depth_counts = Counter(len(p.relative_to(root).parts)-1 for p in files)
    result = {
        'file_count': len(files),
        'extensions': dict(ext_counts),
        'folder_depths': dict(depth_counts),
        'sample_relative_paths': [str(p.relative_to(root)) for p in files[:max(0,args.sample)]],
        'warning': 'This is a structure-only inventory. MRI geometry, labels, patient-level splits and licenses still need validation.'
    }
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
