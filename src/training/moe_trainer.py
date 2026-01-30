import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import time
from typing import Optional, Dict, Any, List, Tuple

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.labels.intraday_labels import IntradayLabels
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
from src.strategies.ml_models.random_forest_intraday import RandomForestIntradayStrategy
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy

def train_moe_models(
    train_start_date: str,
    train_end_date: str,
    output_dir: Path,
    config_path: str = 'config/intraday_config.yaml',
    model_types: List[str] = ['xgb', 'rf', 'lgb']
):
    """
    Trains MoE models (and Global models) on a specific date range.
    """
    print(f"🚀 Training MoE Models ({model_types}) from {train_start_date} to {train_end_date}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data with Date Filtering
    # Note: DataLoader typically loads by period name from config.
    # To support custom dates, we load a larger period (e.g. 'train' + 'valid') and filter manually,
    # OR we need to modify DataLoader to accept dates.
    # Given typical DataLoader structure, it might rely on config. 
    # Workaround: Load all available data (train+valid+test) then slice.
    # Be careful with memory.
    
    print("Loading data...")
    loader = IntradayDataLoader(config_path)
    
    # We load 'train' as base, but for rolling we might need 'valid' data too if expanding window crosses boundary.
    # Safer to load multiple periods or just raw if possible.
    # Let's assume 'train' covers 2000-2018, 'valid' 2019-2021, 'test' 2022-2025.
    # If rolling window needs 2015-2020, we need train+valid.
    
    # Strategy: Load all relevant periods and concat
    chunks = []
    for p in ['train', 'valid', 'test']:
        try:
            chunks.append(loader.get_period_data(p))
        except:
            pass
    
    if not chunks:
        raise ValueError("No data loaded!")
        
    df_full = pd.concat(chunks)
    
    # Filter by date
    # Index level 0 is datetime
    mask = (df_full.index.get_level_values(0) >= pd.Timestamp(train_start_date)) & \
           (df_full.index.get_level_values(0) <= pd.Timestamp(train_end_date))
    df_train = df_full[mask]
    
    if df_train.empty:
        raise ValueError(f"No data found for range {train_start_date} to {train_end_date}")
        
    print(f"Training Data: {len(df_train)} samples")

    # 2. Features
    print("Generating features...")
    alpha = IntradayAlphaFeatures()
    df_train = alpha.generate_all_features(df_train)
    season = SeasonalityFeatures()
    df_train = season.generate_all_features(df_train)
    
    # 3. Labels
    print("Generating labels...")
    labeller = IntradayLabels(config_path)
    df_train = labeller.generate_all_labels(df_train)
    target_col = 'ret_24bar'
    
    # Cleanup NAs
    valid_mask = df_train[target_col].notna() & np.isfinite(df_train[target_col])
    df_train = df_train[valid_mask]
    
    # Identify Features
    label_cols = []
    for cat in labeller.get_label_names().values():
        label_cols.extend(cat)
    ignore_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 
                   'bar_index', 'datetime', 'instrument', 'date', 'time'] + label_cols
    
    feature_cols = [c for c in df_train.columns if c not in ignore_cols]
    
    # 4. Train Models
    # Time Blocks
    blocks = [
        ('0930_1000', time(9, 30), time(10, 0)),
        ('1000_1030', time(10, 0), time(10, 30)),
        ('1030_1200', time(10, 30), time(12, 0)),
        ('1200_1400', time(12, 0), time(14, 0)),
        ('1400_1500', time(14, 0), time(15, 0)),
        ('1500_1600', time(15, 0), time(16, 0)),
    ]
    
    times = df_train.index.get_level_values(0).time

    # Helper for Sample Weights
    def calculate_weights(df_idx, method, min_weight=0.25, half_life=2.0):
        if method == 'none':
            return None
            
        dates = df_idx.get_level_values(0)
        max_date = pd.Timestamp(train_end_date)
        age_years = (max_date - dates).days / 365.25
        # Ensure non-negative
        age_years = np.maximum(age_years, 0)
        
        if method == 'linear':
            max_age = (max_date - pd.Timestamp(train_start_date)).days / 365.25
            if max_age <= 0: return None
            # w = 1 - (age / max_age) * (1 - min)
            decay = 1.0 - min_weight
            weights = 1.0 - (age_years / max_age) * decay
            weights = np.clip(weights, min_weight, 1.0)
            
        elif method == 'exponential':
            # w = 2 ^ (-age / halflife)
            weights = np.power(2, -age_years / half_life)
        else:
            return None
            
        return pd.Series(weights, index=df_idx)

    # Helper for MoE Training
    def train_moe_type(m_type, cls, kwargs):
        print(f"\nTraining MoE {m_type.upper()}...")
        moe_dir = output_dir / f'moe_{m_type}'
        moe_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}
        
        # Determine weighting from config (loaded via loader logic or raw helper)
        # We need raw config to access 'training' section which might not be in the Strategy/Loader config object
        # Quick hack: Load raw yaml
        import yaml
        with open(config_path) as f:
            raw_conf = yaml.safe_load(f)
        
        weight_conf = raw_conf.get('training', {}).get('sample_weighting', {})
        w_method = weight_conf.get('method', 'none')
        w_min = weight_conf.get('linear_min_weight', 0.25)
        w_halflife = weight_conf.get('exponential_half_life_years', 2.0)
        
        print(f"  Weighting Method: {w_method}")

        for name, start_t, end_t in blocks:
            # print(f"  Block: {name}") 
            block_mask = (times >= start_t) & (times < end_t)
            df_block = df_train[block_mask]
            
            if len(df_block) < 500: # relaxed check
                print(f"    ⚠️ Skip {name} (n={len(df_block)})")
                continue
                
            model = cls(kwargs)
            # Handle potential feature mismatch or strict mode
            # We strictly pass feature cols
            X = df_block[feature_cols].copy()
            y = df_block[target_col].copy()
            
            # Calculate weights for this block
            weights = calculate_weights(df_block.index, w_method, w_min, w_halflife)
            if weights is not None:
                print(f"    Weights (min/mean/max): {weights.min():.2f} / {weights.mean():.2f} / {weights.max():.2f}")
            
            # Additional safety: fillna for tree models
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            model.fit(X, y, sample_weight=weights)
            
            fname = f"{name}.joblib"
            # Some wrappers store in .model, some are the model
            obj_to_save = model.model if hasattr(model, 'model') else model
            joblib.dump(obj_to_save, moe_dir / fname)
            manifest[start_t.strftime("%H:%M")] = fname
            
        with open(moe_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f)

    # Configs
    rf_config = {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 20, 'n_jobs': -1}
    xgb_config = {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'n_jobs': -1}
    lgb_config = {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31, 'n_jobs': -1}

    if 'xgb' in model_types:
        train_moe_type('xgb', XGBoostIntradayStrategy, xgb_config)
        
    if 'rf' in model_types:
        train_moe_type('rf', RandomForestIntradayStrategy, rf_config)
        
    if 'lgb' in model_types:
        train_moe_type('lgb', LightGBMIntradayStrategy, lgb_config)
            
    print(f"✅ Training Complete for {train_end_date}")
