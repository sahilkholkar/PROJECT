Anomaly Detection in Surveillance Videos using Convolutional Autoencoder
•	Built a deep learning-based surveillance anomaly detection system using PyTorch, leveraging Convolutional Autoencoders and SSIM loss to identify abnormal human actions (e.g., fall, hit, sneak).
•	Utilized OpenCV for frame extraction and preprocessing from CCTV footage; converted video clips into grayscale image datasets for efficient processing.
•	Applied torchvision.transforms, DataLoader, and custom PyTorch Dataset classes for scalable data pipeline and efficient model training.
•	Trained the model using noisy input reconstruction with SSIM loss (from pytorch-msssim) instead of traditional MSE to better preserve perceptual image quality.
•	Evaluated model using ROC AUC, confusion matrix, and error distribution plots with Matplotlib, Seaborn, and scikit-learn.
•	Visualized and compared original vs reconstructed frames to analyze model accuracy in real-time anomaly detection.
•	Addressed challenges like overlapping error values for normal and anomaly samples by using noise injection and 95th percentile thresholding for classification.
•	Tech Stack: PyTorch, OpenCV, pytorch-msssim, Torchvision,  Seaborn, scikit-learn, NumPy, PIL, Google Colab, Python
