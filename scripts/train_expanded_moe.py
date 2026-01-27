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
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
from src.strategies.ml_models.random_forest_intraday import RandomForestIntradayStrategy

def train_expanded_moe():
    print("🚀 Starting Expanded MoE Training (XGB & RF)...")
    
    # 1. Load Data
    print("Loading training data...")
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_train = loader.get_period_data('train')
    
    # 2. Features
    print("Generating features...")
    alpha = IntradayAlphaFeatures()
    df_train = alpha.generate_all_features(df_train)
    season = SeasonalityFeatures()
    df_train = season.generate_all_features(df_train)
    
    # 3. Labels
    print("Generating labels...")
    labeller = IntradayLabels('config/intraday_config.yaml')
    df_train = labeller.generate_all_labels(df_train)
    target_col = 'ret_24bar'
    
    # Prepare Data
    valid_mask = df_train[target_col].notna() & np.isfinite(df_train[target_col])
    df_train = df_train[valid_mask]
    
    label_cols = []
    for cat in labeller.get_label_names().values():
        label_cols.extend(cat)
    ignore_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 
                   'bar_index', 'datetime', 'instrument', 'date', 'time'] + label_cols
    
    feature_cols = [c for c in df_train.columns if c not in ignore_cols]
    print(f"Features: {len(feature_cols)}, Samples: {len(df_train)}")

    # Time Blocks
    blocks = [
        ('0930_1000', time(9, 30), time(10, 0)),
        ('1000_1030', time(10, 0), time(10, 30)),
        ('1030_1200', time(10, 30), time(12, 0)),
        ('1200_1400', time(12, 0), time(14, 0)),
        ('1400_1500', time(14, 0), time(15, 0)),
        ('1500_1600', time(15, 0), time(16, 0)),
    ]

    # --- Train Global RF ---
    print("\n🌲 Training Global Random Forest...")
    rf_global_dir = Path.cwd() / 'results' / 'models' / 'randomforest_global'
    rf_global_dir.mkdir(parents=True, exist_ok=True)
    
    # Use config from implementation plan
    rf_config = {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 20, 'n_jobs': -1}
    
    rf_global = RandomForestIntradayStrategy(rf_config)
    rf_global.fit(df_train[feature_cols], df_train[target_col])
    rf_global.save_model(str(rf_global_dir))
    
    # --- Train MoE XGB ---
    print("\n🚀 Training MoE XGBoost Models...")
    xgb_out_dir = Path.cwd() / 'results' / 'models' / 'moe_xgb'
    xgb_out_dir.mkdir(parents=True, exist_ok=True)
    xgb_manifest = {}
    
    # Logic to filter by time
    times = df_train.index.get_level_values(0).time
    
    for name, start_t, end_t in blocks:
        print(f"  Block: {name}")
        mask = (times >= start_t) & (times < end_t)
        df_block = df_train[mask]
        
        if len(df_block) < 1000:
            print("    ⚠️ Too few samples")
            continue
            
        xgb_model = XGBoostIntradayStrategy({'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'n_jobs': -1})
        xgb_model.fit(df_block[feature_cols], df_block[target_col])
        
        # Save raw model object for MoE strategy to load
        fname = f"{name}.joblib"
        joblib.dump(xgb_model.model, xgb_out_dir / fname)
        xgb_manifest[start_t.strftime("%H:%M")] = fname
        
    with open(xgb_out_dir / 'manifest.json', 'w') as f:
        json.dump(xgb_manifest, f)
        
    # --- Train MoE RF ---
    print("\n🌲 Training MoE Random Forest Models...")
    rf_out_dir = Path.cwd() / 'results' / 'models' / 'moe_rf'
    rf_out_dir.mkdir(parents=True, exist_ok=True)
    rf_manifest = {}
    
    for name, start_t, end_t in blocks:
        print(f"  Block: {name}")
        mask = (times >= start_t) & (times < end_t)
        df_block = df_train[mask]
        
        if len(df_block) < 1000: continue
            
        rf_model = RandomForestIntradayStrategy(rf_config)
        rf_model.fit(df_block[feature_cols], df_block[target_col])
        
        fname = f"{name}.joblib"
        joblib.dump(rf_model.model, rf_out_dir / fname)
        rf_manifest[start_t.strftime("%H:%M")] = fname

    with open(rf_out_dir / 'manifest.json', 'w') as f:
        json.dump(rf_manifest, f)
        
    print("\n✅ All Expanded MoE Models Trained!")

if __name__ == "__main__":
    train_expanded_moe()
