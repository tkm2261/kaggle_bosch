rm stack*csv
python xgb_split.py && \
cp list_xgb_model.pkl list_xgb_model_1.pkl && \
cp stack*csv tmp/    
python xgb_split.py
