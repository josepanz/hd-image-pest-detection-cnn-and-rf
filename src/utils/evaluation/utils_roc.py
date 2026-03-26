import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_combined_roc(roc_files, labels):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.figure(figsize=(8, 6))

    for file, label in zip(roc_files, labels):
        data = np.load(file)
        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{label} (AUC = {data['auc']:.4f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC comparativas")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results/combined_roc_curve_{timestamp}.png")
    plt.show()
    plt.close('all')

def main():
  parser = argparse.ArgumentParser(description="Plotea curvas ROC comparativas desde archivos guardados")
  args = parser.parse_args()
  roc_files = [
    r"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\CNN\MULTIESPECTRAL\binary_crossentropy\0.5\ROC_data_best_model_final_MULTIESPECTRAL_binary_crossentropy.keras_20260325_1152_t0.5.npz",
    r"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\CNN\MULTIESPECTRAL\focal_loss\0.7\ROC_data_best_model_final_MULTIESPECTRAL_focal_loss.keras_20260325_1159_t0.7.npz",
    
    r"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\CNN\RGB\binary_crossentropy\0.7\ROC_data_best_model_final_RGB_binary_crossentropy.keras_20260325_1509_t0.7.npz",
    r"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\CNN\RGB\focal_loss\0.7\ROC_data_best_model_final_RGB_focal_loss.keras_20260325_1445_t0.7.npz",
    
    r"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RANDOM_FOREST\MULTIESPECTRAL\0.7\ROC_data_report_table_best_model_RANDOM_FOREST_MULTIESPECTRAL_20260325_1634_t0.7.npz",
    r"C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RANDOM_FOREST\RGB\0.7\ROC_data_report_table_best_model_RANDOM_FOREST_RGB_20260325_1643_t0.7.npz"
  ]

  labels = [
      "CNN Multiespectral BC",
      "CNN Multiespectral FL",

      "CNN RGB BC",
      "CNN RGB FL",

      "RF Multiespectral",
      "RF RGB"
  ]
  plot_combined_roc(roc_files, labels)

if __name__ == "__main__":
  main()