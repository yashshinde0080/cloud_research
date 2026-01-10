
import logging
import numpy as np
import pandas as pd
from prediction import HybridPredictor
from config.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference():
    """Load trained model and predict for a sample machine"""
    
    # 1. Initialize Predictor
    predictor = HybridPredictor(
        arima_order=config.prediction.arima_order,
        lstm_lookback=config.prediction.lstm_lookback
    )
    
    # 2. Load trained models
    logger.info(f"Loading models from {config.models_path}...")
    predictor.load_models(config.models_path)
    
    if not predictor.models:
        logger.error("No models found! Run 'python main.py --data-path data/raw' first to train.")
        return

    # 3. Pick a machine to predict
    machine_id = list(predictor.models.keys())[0]
    model_info = predictor.models[machine_id]
    logger.info(f"Selected Machine: {machine_id}")
    logger.info(f"Workload Type: {model_info['workload_type']}")
    logger.info(f"Strategy: {model_info['strategy']}")
    
    # 4. Generate Prediction (Forecasting future steps)
    steps = 5
    logger.info(f"Generating forecast for next {steps} steps...")
    
    # Note: In a real scenario, you'd pass the recent history (last_sequence) for LSTM.
    # Here we let the predictor handle it (it uses internal state or defaults if history missing)
    predictions = predictor.predict_machine(machine_id, steps=steps)
    
    print("\n" + "="*40)
    print(f"PREDICTIONS FOR {machine_id}")
    print("="*40)
    print(f"Next {steps} time steps:")
    for i, p in enumerate(predictions):
        print(f"Step +{i+1}: {p:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_inference()
