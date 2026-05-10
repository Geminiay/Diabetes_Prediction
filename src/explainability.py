import shap
import matplotlib.pyplot as plt

def generate_shap_report(model, X_test, output_path="models/shap_summary.png"):
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    plt.figure(figsize=(10, 6))

    if isinstance(shap_values, list):
        target_shap_values = shap_values[1]
    else:
        if len(shap_values.shape) == 3:
            target_shap_values = shap_values[:, :, 1]
        else:
            target_shap_values = shap_values

    shap.summary_plot(target_shap_values, X_test, show=False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 
    print(f"[SUCCESS] SHAP summary plot saved to: {output_path}")