# AI Crowd Density Estimation Web App

An AI-powered web application for estimating crowd density from images using deep learning.

#About the Project

This project uses CSRNet, a deep learning-based crowd counting model, to estimate the number of people present in an uploaded image.

The application also generates a density heatmap to visually represent areas with higher crowd concentration.

#Features

- Upload a crowd image
- Automatic image processing
- AI-based crowd counting
- Crowd density classification
- Density heatmap generation
- Estimated processing time
- Web-based interface
- Image preview before analysis

# Technology Stack

- **Python**
- **PyTorch**
- **CSRNet**
- **Flask**
- **HTML**
- **CSS**
- **JavaScript**
- **NumPy**
- **Pillow**
- **Matplotlib**

# How It Works

1. The user uploads an image containing people.
2. The Flask application receives and saves the image.
3. The image is processed and passed to the CSRNet model.
4. The model generates a density map.
5. The density map is summed to estimate the number of people.
6. The application classifies the crowd as low, medium, or high density.
7. A heatmap is generated and displayed on the result page.

#Project Structure

```text
AI_Crowd_Density_Estimation/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── crowd_model.py
│   ├── csrnet.py
│   └── weights.pth
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── utils/
    └── image_processing.py
```

# Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

#Running the Application

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Model

The application uses CSRNet (Congested Scene Recognition Network) for crowd density estimation.

The model produces a density map, which is used to estimate the total number of people in the image.

## Author

Sanzida Ahmed Aroni

Bachelor of Science in Computer Science

Thesis related :

AI Crowd Density Estimation Web Application
