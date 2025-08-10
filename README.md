# Fruit Ripeness Detection using SVM

This project uses classical computer vision techniques and a Support Vector Machine (SVM) to classify the type and ripeness of fruits from an image.

## How to Run

1.  **Add Data:** Populate the `dataset/` folder with your images, organized into subfolders by class (e.g., `Apple_Ripe`).
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Train the Model:** `python train_model.py`
4.  **Run the Application:** `python app.py`

<!-- FOR GIT CLONE USERS -->
# Fruit Ripeness Detection using SVMS

This project uses computer vision and an SVM to classify fruit ripeness.

## Dataset Setup

**This repository does not include the dataset due to its size.**

To run this project, you must create your own dataset with the following structure inside the `dataset/` folder:

dataset/
├── Apple_Ripe/
│   ├── image01.jpg
│   └── ...
├── Apple_Unripe/
│   ├── image01.jpg
│   └── ...
├── Banana_Ripe/
│   ├── image01.jpg
│   └── ...
└── (etc. for all your classes)

## How to Run

1.  **Create the dataset** as described above.
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Train the Model:** `python train_model.py`
4.  **Run the Application:** `python app.py`