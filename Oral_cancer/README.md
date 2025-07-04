# Oral Cancer Risk Assessment Web Application

A web-based application for oral cancer detection using gene expression data with interactive visualizations and a comprehensive glossary of medical terms. The application features a genetics-themed interface and uses machine learning to predict cancer risk based on user input.

## Features

- Real-time predictions using machine learning
- Interactive visualizations of cancer statistics
- Comprehensive glossary of medical and genetic terms
- Modern, genetics-themed user interface
- Responsive design for all devices

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Oral_cancer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Train the machine learning model:
```bash
python model/train_model.py
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Navigate to the home page to access the risk assessment form
2.upload the file
3. Submit the file
4. Explore the visualizations page for statistical insights
5. Visit the glossary to learn about medical and genetic terms

## Technical Details

- Backend: Flask (Python)
- Frontend: HTML5, CSS3, JavaScript
- Visualization: Plotly.js
- Machine Learning: scikit-learn
- Model: Random Forest Classifier
- Styling: Custom CSS with genetics theme

## Important Notes

- This tool is for educational purposes only
- Consult healthcare professionals for medical advice
- The predictions are based on statistical models and should not be used as a definitive diagnosis



## License

This project is licensed under the MIT License - see the LICENSE file for details. 
