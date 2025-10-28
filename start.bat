@echo off

REM echo Resample all input rasters to the reference grid/resolution
REM python resample_to_reference.py

echo Optional conversion/visualization util (use only if needed)
python convert.py

echo Sample pixels, filter NoData/clouds, build features + label -> final_dataset_from_resampled_sampled.csv
python extract_labeled_dataset_from_resampled_sampled.py

echo Normalize numeric features, keep OHE as-is; export dnn_ready_dataset.csv + feature_columns.csv + min_max_values.csv
python prepare_ddn_dataset.py


echo Train residual DNN; save best checkpoint & metrics
python DNN_Train.py

echo Tile-wise DNN inference to GeoTIFF probability + RGB visualization
python DNN_predict_to_raster.py

echo DNN threshold sweep (F1) + SHAP summary & per-feature importances CSV
python DNN_threshold_and_shap.py


echo Train XGBoost with early stopping; export model JSON + reports
python XGBoost_Train.py

echo Tile-wise XGBoost inference (uses feature_columns/min_max)
python XGBoost_predict_to_raster.py

echo XGBoost threshold sweep (F1) + SHAP summary & CSV
python XGBoost_threshold_and_shap.py


echo Train KAN residual MLP; save best model + report
python KAN_Train.py

echo Tile-wise KAN inference to GeoTIFF probability + RGB visualization
python KAN_predict_to_raster.py

echo KAN thresholding / explainability (if available)
python KAN_threshold_and_shap.py


echo Cross-model Spearman rank correlation of feature importances
python cross_model_spearman.py

echo Done.