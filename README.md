Axillary Lymph Node Detection Using Machine Learning
Overview
This project focuses on the early detection of axillary lymph node metastasis in mammography images using machine learning and deep learning techniques. By leveraging radiomics feature extraction and image segmentation, the system aims to enhance cancer diagnosis accuracy. The approach integrates both traditional machine learning and deep learning models, enabling more precise detection of lymph node abnormalities.

Features
Radiomics-Based Feature Extraction: Extracts quantitative imaging biomarkers from mammograms.
Deep Learning & Traditional Models: Combines CNNs, SVMs, and Random Forest for optimal performance.
Image Segmentation: Detects and isolates lymph nodes in mammograms.
Automated Classification: Differentiates between benign and malignant lymph nodes.
Technologies Used
Programming Languages: Python, MATLAB
Machine Learning Frameworks: TensorFlow, PyTorch, Scikit-Learn
Medical Image Processing: OpenCV, SimpleITK
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Methodology
Data Preprocessing: Enhancing mammography images using noise reduction and normalization.
Feature Extraction: Applying radiomics techniques to obtain relevant biomarkers.
Segmentation: Identifying axillary lymph nodes using deep learning-based image segmentation.
Model Training: Comparing traditional ML classifiers (SVM, Random Forest) with deep learning models (CNNs, U-Net).
Evaluation: Assessing model performance using accuracy, AUC-ROC, precision, and recall metrics.
Installation & Usage
Prerequisites
Ensure you have the following installed:

Python 3.8+
MATLAB
TensorFlow, PyTorch
OpenCV, SimpleITK
Scikit-Learn, Pandas, NumPy
Run the Project
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/axillary-lymph-node-detection.git
cd axillary-lymph-node-detection
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the preprocessing and feature extraction:
bash
Copy
Edit
python preprocess.py
Train the model:
bash
Copy
Edit
python train.py
Evaluate and visualize results:
bash
Copy
Edit
python evaluate.py
Results
The deep learning model achieved X% accuracy, outperforming traditional ML approaches.
The use of radiomics features improved classification performance.
The segmentation approach reduced false positives and improved early-stage detection reliability.
Future Work
Enhancing Model Generalization: Training on larger and more diverse datasets.
Integrating Multi-Modal Data: Combining mammography with ultrasound and MRI.
Deploying as a Web-Based Tool: Creating an interactive visualization dashboard.
Contributors
Hossein Fathollahian – PhD Researcher in Computer Vision & AI
University of Illinois Chicago – Electronic Visualization Laboratory (EVL)
License
This project is open-source and available under the MIT License.
