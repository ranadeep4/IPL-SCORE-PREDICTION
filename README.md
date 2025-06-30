# IPL Score Prediction

This project predicts IPL cricket match scores using a neural network model built with Keras and TensorFlow. It includes EDA, model training, and a web app for predictions.

## Project Structure
- `notebooks/EDA_and_model_dev.ipynb`: EDA and initial exploration
- `src/`: Source code for training, prediction, and utilities
- `saved_models/`: Trained model and encoders
- `app/`: Streamlit app for user interaction

## Setup
1. Create and activate the conda environment:
   ```
   conda create -n ipl python=3.10
   conda activate ipl
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run training:
   ```
   python src/train.py
   ```
4. Run the app:
   ```
   streamlit run app/app.py
   ```

## Data
Place your `ipl_data.csv` in the `content/` directory or update the path in the scripts. 