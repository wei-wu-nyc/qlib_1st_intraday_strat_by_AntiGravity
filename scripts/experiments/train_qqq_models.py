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
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
from src.strategies.ml_models.random_forest_intraday import RandomForestIntradayStrategy

def train_qqq_models():
    print("🚀 Starting QQQ-Only Training (LGB, XGB, RF)...")
    
    # 1. Load Data (QQQ Only)
    print("Loading training data (QQQ Only)...")
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_train = loader.get_period_data('train', symbols=['QQQ'])
    
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

    # Define Base Output Directory
    base_out_dir = Path.cwd() / 'results' / 'models_qqq_only'
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Time Blocks for MoE
    blocks = [
        ('0930_1000', time(9, 30), time(10, 0)),
        ('1000_1030', time(10, 0), time(10, 30)),
        ('1030_1200', time(10, 30), time(12, 0)),
        ('1200_1400', time(12, 0), time(14, 0)),
        ('1400_1500', time(14, 0), time(15, 0)),
        ('1500_1600', time(15, 0), time(16, 0)),
    ]

    # --- Train Global Models ---
    print("\n🌍 Training Global Models...")
    
    # Global LGB
    print("  LightGBM (Global)...")
    lgb_global = LightGBMIntradayStrategy({'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 31})
    lgb_global.fit(df_train[feature_cols], df_train[target_col])
    lgb_global.save_model(str(base_out_dir / 'lgb_global'))
    
    # Global XGB
    print("  XGBoost (Global)...")
    xgb_global = XGBoostIntradayStrategy({'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'n_jobs': -1})
    xgb_global.fit(df_train[feature_cols], df_train[target_col])
    # manually save logic since strategy class expects directory for general use
    # but here we are customized. The strategy .save_model usually saves to a dir.
    # Let's check implementation of save_model. Assuming standard behavior:
    xgb_global.save_model(str(base_out_dir / 'xgb_global'))
    
    # Global RF
    print("  Random Forest (Global)...")
    rf_config = {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 20, 'n_jobs': -1}
    rf_global = RandomForestIntradayStrategy(rf_config)
    rf_global.fit(df_train[feature_cols], df_train[target_col])
    rf_global.save_model(str(base_out_dir / 'rf_global'))
    
    # --- Train MoE Models ---
    print("\n🧠 Training MoE Models (Time-Specific)...")
    
    # Prepare Dirs
    moe_lgb_dir = base_out_dir / 'moe_lgb'
    moe_xgb_dir = base_out_dir / 'moe_xgb'
    moe_rf_dir = base_out_dir / 'moe_rf'
    for d in [moe_lgb_dir, moe_xgb_dir, moe_rf_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    times = df_train.index.get_level_values(0).time
    manifest_lgb = {}
    manifest_xgb = {}
    manifest_rf = {}
    
    for name, start_t, end_t in blocks:
        print(f"  Block: {name}")
        mask = (times >= start_t) & (times < end_t)
        df_block = df_train[mask]
        
        if len(df_block) < 500: # Lower threshold for single asset
            print("    ⚠️ Too few samples")
            continue
            
        # LGB MoE
        lgb_m = LightGBMIntradayStrategy({'n_estimators': 150, 'learning_rate': 0.05, 'num_leaves': 20})
        lgb_m.fit(df_block[feature_cols], df_block[target_col])
        fname = f"{name}.joblib"
        joblib.dump(lgb_m.model, moe_lgb_dir / fname)
        manifest_lgb[start_t.strftime("%H:%M")] = fname
        
        # XGB MoE
        xgb_m = XGBoostIntradayStrategy({'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'n_jobs': -1})
        xgb_m.fit(df_block[feature_cols], df_block[target_col])
        fname = f"{name}.joblib"
        joblib.dump(xgb_m.model, moe_xgb_dir / fname)
        manifest_xgb[start_t.strftime("%H:%M")] = fname
        
        # RF MoE
        rf_m = RandomForestIntradayStrategy(rf_config)
        rf_m.fit(df_block[feature_cols], df_block[target_col])
        fname = f"{name}.joblib"
        joblib.dump(rf_m.model, moe_rf_dir / fname)
        manifest_rf[start_t.strftime("%H:%M")] = fname

    # Save Manifests
    with open(moe_lgb_dir / 'manifest.json', 'w') as f: json.dump(manifest_lgb, f)
    with open(moe_xgb_dir / 'manifest.json', 'w') as f: json.dump(manifest_xgb, f)
    with open(moe_rf_dir / 'manifest.json', 'w') as f: json.dump(manifest_rf, f)
        
    print(f"\n✅ All QQQ-Only Models Trained! Results in {base_out_dir}")

if __name__ == "__main__":
    train_qqq_models()
