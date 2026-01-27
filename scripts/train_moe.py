import pandas as pd
import numpy as np
import sys
import json
import joblib
from pathlib import Path
from datetime import time

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.labels.intraday_labels import IntradayLabels
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy

def train_moe_models():
    print("🚀 Starting MoE Model Training...")
    
    # 1. Load Data (Train set only)
    print("Loading training data...")
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_train = loader.get_period_data('train')
    
    # 2. Features
    print("Generating features...")
    alpha = IntradayAlphaFeatures()
    df_train = alpha.generate_all_features(df_train)
    season = SeasonalityFeatures()
    df_train = season.generate_all_features(df_train)
    
    # 3. Generate Labels (Target)
    print("Generating labels (Target: ret_24bar)...")
    labeller = IntradayLabels('config/intraday_config.yaml')
    df_train = labeller.generate_all_labels(df_train)
    
    target_col = 'ret_24bar'
    if target_col not in df_train.columns:
        raise ValueError(f"Target column {target_col} not found in labels!")
        
    # Define columns to exclude from features
    label_cols = []
    label_names = labeller.get_label_names()
    for cat in label_names.values():
        label_cols.extend(cat)
        
    ignore_cols = [
        'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 
        'bar_index', 'datetime', 'instrument', 'date', 'time'
    ] + label_cols
    
    # 4. Define Time Blocks
    # Format: (name, start_time, end_time) - half-open interval [start, end)
    blocks = [
        ('0930_1000', time(9, 30), time(10, 0)),
        ('1000_1030', time(10, 0), time(10, 30)),
        ('1030_1200', time(10, 30), time(12, 0)),
        ('1200_1400', time(12, 0), time(14, 0)),
        ('1400_1500', time(14, 0), time(15, 0)),
        ('1500_1600', time(15, 0), time(16, 0)),
    ]
    
    output_dir = Path.cwd() / 'results' / 'models' / 'moe'
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    
    # 5. Train Models
    for name, start_t, end_t in blocks:
        print(f"\nTraining Model: {name} ({start_t} - {end_t})")
        
        # Filter Data
        # df index is MultiIndex (datetime, instrument)
        times = df_train.index.get_level_values(0).time
        
        # Handling [start, end) logic
        # For 15:00-16:00, ensure we include 16:00 if data exists, but markets close at 16:00
        # Data usually ends at 15:55 or 16:00.
        mask = (times >= start_t) & (times < end_t)
        df_block = df_train[mask].copy()
        
        print(f"  Samples: {len(df_block)}")
        
        if len(df_block) < 1000:
            print("  ⚠️ Warning: Too few samples, skipping!")
            continue
            
        # Prepare X and y
        # Filter valid target
        valid_mask = df_block[target_col].notna() & np.isfinite(df_block[target_col])
        df_block = df_block[valid_mask]
        
        if len(df_block) == 0:
             print("  ⚠️ No valid targets!")
             continue

        y = df_block[target_col]
        # Identify feature columns (all columns not in ignore list)
        feature_cols = [c for c in df_block.columns if c not in ignore_cols]
        X = df_block[feature_cols]
        
        print(f"  Features: {len(feature_cols)}, Target: {target_col}")
            
        # Train
        strategy = LightGBMIntradayStrategy({})
        strategy.fit(X, y)
        
        # Save
        model_filename = f"{name}.joblib"
        model_path = output_dir / model_filename
        joblib.dump(strategy.model, model_path)
        print(f"  ✅ Saved to {model_filename}")
        
        key = start_t.strftime("%H:%M")
        manifest[key] = model_filename

    # 6. Save Manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"\n✅ Manifest saved to {manifest_path}")

if __name__ == "__main__":
    train_moe_models()
