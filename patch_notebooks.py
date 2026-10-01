import json

for nb_path in ['CIC/05_Model_Evaluation.ipynb', 'LUFlow/05_Model_Evaluation.ipynb']:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            new_src = []
            for line in src:
                if 'from sklearn.metrics import plot_confusion_matrix' in line:
                    new_src.append("try:\n")
                    new_src.append("    from sklearn.metrics import plot_confusion_matrix\n")
                    new_src.append("except ImportError:\n")
                    new_src.append("    from sklearn.metrics import ConfusionMatrixDisplay\n")
                    new_src.append("    def plot_confusion_matrix(estimator, X, y, ax=None, **kwargs):\n")
                    new_src.append("        return ConfusionMatrixDisplay.from_estimator(estimator, X, y, ax=ax, **kwargs)\n")
                    modified = True
                else:
                    new_src.append(line)
            cell['source'] = new_src
    
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {nb_path} for modern scikit-learn compatibility.")
