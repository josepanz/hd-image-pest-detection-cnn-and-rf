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
            label=f"{label} (AUC = {data['auc']:.2f})"
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
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\MULTIESPECTRAL\ROC_data_best_model_val_loss_MULTIESPECTRAL_binary_crossentropy_20260102_1155_t0.45.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\MULTIESPECTRAL\ROC_data_best_model_val_loss_MULTIESPECTRAL_binary_crossentropy_20260102_1156_t0.5.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\MULTIESPECTRAL\ROC_data_best_model_val_loss_MULTIESPECTRAL_binary_crossentropy_20260102_1152_t0.7.npz",

    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\MULTIESPECTRAL\ROC_data_best_model_val_loss_MULTIESPECTRAL_focal_loss_20260102_1316_t0.45.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\MULTIESPECTRAL\ROC_data_best_model_val_loss_MULTIESPECTRAL_focal_loss_20260102_1320_t0.5.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\MULTIESPECTRAL\ROC_data_best_model_val_loss_MULTIESPECTRAL_focal_loss_20260102_1324_t0.7.npz",
    
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RGB\ROC_data_best_model_val_loss_RGB_binary_crossentropy_20260102_1336_t0.45.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RGB\ROC_data_best_model_val_loss_RGB_binary_crossentropy_20260102_1338_t0.5.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RGB\ROC_data_best_model_val_loss_RGB_binary_crossentropy_20260102_1340_t0.7.npz",

    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RGB\ROC_data_best_model_val_loss_RGB_focal_loss_20260102_1327_t0.45.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RGB\ROC_data_best_model_val_loss_RGB_focal_loss_20260102_1333_t0.5.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RGB\ROC_data_best_model_val_loss_RGB_focal_loss_20260102_1334_t0.7.npz",
    
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RANDOM_FOREST\ROC_data_report_table_best_model_RANDOM_FOREST_MULTIESPECTRAL_20260102_1341_t0.5.npz",
    "C:\workspace\hd-image-pest-detection-cnn-and-rf\src\pest_detection_models\evaluation_results\RANDOM_FOREST\ROC_data_report_table_best_model_RANDOM_FOREST_RGB_20260102_1344_t0.5.npz"
  ]

  labels = [
      "CNN Multiespectral BC 0.45",
      "CNN Multiespectral BC 0.50",
      "CNN Multiespectral BC 0.70",

      "CNN Multiespectral FL 0.45",
      "CNN Multiespectral FL 0.50",
      "CNN Multiespectral FL 0.70",

      "CNN RGB BC 0.45",
      "CNN RGB BC 0.50",
      "CNN RGB BC 0.70",

      "CNN RGB FL 0.45",
      "CNN RGB FL 0.50",
      "CNN RGB FL 0.70",

      "RF Multiespectral",
      "RF RGB"
  ]
  plot_combined_roc(roc_files, labels)

if __name__ == "__main__":
  main()