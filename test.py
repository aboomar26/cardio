import torch
import joblib
import numpy as np
import pandas as pd
import fastapi
import pydantic
import xgboost
import sklearn

print("torch:", torch.__version__)
print("joblib:", joblib.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("fastapi:", fastapi.__version__)
print("pydantic:", pydantic.VERSION)
print("xgboost:", xgboost.__version__)
print("scikit-learn:", sklearn.__version__)
