import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
# Load historical data in a dataframe
# Load new data in a dataframe
# Check if any of the new data is a duplicate and already exists in the historical data
# Filter the new data from these duplicates
# Concatenate the filterated data to the historical data
# Do the same data cleaning and prcoessing on the whole data
# Retrain the model and replace the old model

# We considered doing the cleaning and processing on the new data only, but what if the supposedly new data has duplicated records with the historical data?

# Data preprocessing function:
# HISTORICAL_DATA = r"data\historical_data.csv"
# NEW_DATA = r"data\new_data.csv"
# TRAINING_DATA = r'data\training_data.csv'
def retraining(historical_data:str, new_data:str, old_model:str) -> dict:
    new_df = pd.read_csv(new_data, dtype={'Customer ID': str})
    try:
        # This line will cause an error if no historical data exist at all.
        historical_df = pd.read_csv(historical_data, dtype={'Customer ID': str})

        # Crash the script if their # columns isn't equal
        if new_df.shape[1] != historical_df.shape[1]:
            raise ValueError(f"Dataframes' shapes should be EQUAL!\nIncorrect input!")
    
        # Concatenate both dataframes and update the historical_data_file
        data_df = pd.concat([new_df, historical_df], ignore_index=True)
        data_df.drop_duplicates()
        data_df.to_csv(historical_data, index= False)
    except Exception:
        historical_df = new_df
        historical_df.to_csv(historical_data, index= False)
        data_df = new_df
    # Clean:
    data_df.rename(columns={"Price":"UnitPrice", "Customer ID":"CustomerID"}, inplace=True)
    data_df['CustomerID'] = data_df['CustomerID'].str.split(pat='.', expand=True)[0]
    data_df['InvoiceDate'] = data_df['InvoiceDate'].str.split(pat=' ', expand=True)[0]
    data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])
    data_df["CustomerID"].fillna("Guest", inplace=True)
    data_df = data_df.drop_duplicates()
    data_df = data_df[(data_df["Quantity"] >= 1) & (data_df["UnitPrice"] > 0)]
    data_df = data_df[~data_df['Invoice'].str.contains('C', na=False)]
    print(data_df.shape) # UNTIL HERE, EVERYTHING IS GOOD
    data_df["Revenue"] = (data_df["Quantity"] * data_df["UnitPrice"]).round(2)
    data_df["isGuest"] = data_df["CustomerID"] == "Guest"
    data_df["Month"] = data_df["InvoiceDate"].dt.month
    data_df["Day"] = data_df["InvoiceDate"].dt.day
    # Removing outliers
    lower_limit = data_df["Quantity"].quantile(0.05)  # 5th percentile
    upper_limit = data_df["Quantity"].quantile(0.95)  # 95th percentile
    data_df["Quantity"] = np.where(data_df["Quantity"] > upper_limit, upper_limit, data_df["Quantity"])
    data_df["Quantity"] = np.where(data_df["Quantity"] < lower_limit, lower_limit, data_df["Quantity"])
    # Category Dictionary:
    CATEGORY_KEYWORDS = {
        'Seasonal': ['christmas', 'xmas', 'easter', 'santa', 'snowflake', 'reindeer', 'egg', 'advent', 'village', 'snowmen', 'toadstool', 'cacti', 'halloween', 'decoupage', 'ribbon', 'flock', 'stocking', 'bauble', 'bunny', 'chick', 'tree decoration', 'easter hen', 'christmas ball', 'jingle bell', 'tinsel', 'ornament', 'felt decoration', 'garland stars', 'bauble set', 'icicle lights', 'craft decoration', 'paper chain'],
        'Home Decor': ['knob', 't-light', 'holder', 'wicker', 'mirror', 'wall art', 'heart', 'frame', 'wreath', 'clock', 'cushion', 'doily', 'doiley', 'vase', 'shelf', 'tile', 'hook', 'planter', 'chandelier', 'curio', 'photo frame', 'trellis', 'antique', 'vintage', 'parisienne', 'edwardian', 'chalkboard', 'coaster', 'placemat', 'candlestick', 'stand', 'cornice', 'cloche', 'frieze', 'quilt', 'trinket tray', 'photo clip', 'lattice', 'shelf unit', 'sign', 'garland', 'curtain', 'plaque', 'mirror ball', 'doormat', 'wall mirror', 'votive holder', 'candlestick candle', 'candlepot', 'photo holder', 'wall plaque', 'pillow filler', 'decorative tray', 'wall shelf', 'ornamental frame', 'word block', 'diner clock', 'decorative hook', 'record frame'],
        'Kitchenware': ['jug', 'pantry', 'chopping board', 'spoon', 'mug', 'plate', 'enamel', 'bowl', 'tray', 'jampot', 'thermo', 'utensil', 'tin', 'towel', 'mould', 'ladle', 'brush', 'corer', 'cannister', 'frying pan', 'cakestand', 'teapot', 'colander', 'toastrack', 'butter dish', 'sugar bowl', 'milk jug', 'napkin', 'pepper', 'espresso', 'pizza dish', 'mixing bowl', 'teatime', 'bread basket', 'coffee pot', 'sugar caddy', 'coffee container', 'biscuit bin', 'teacup', 'pudding bowl', 'oven glove', 'serving spoon', 'kitchen towel', 'measuring spoon', 'cake plate'],
        'Stationery': ['card', 'pencil', 'notepad', 'shopping list', 'wrap', 'tissue', 'chalk', 'stamp', 'book mark', 'book', 'exercise book', 'ruler', 'magnet', 'pen', 'stencil', 'paperweight', 'calendar', 'tape', 'sharpener', 'rubber', 'notebook', 'pens', 'sticky', 'eraser', 'calculator', 'memo pad', 'journal', 'sticker sheet', 'to do list', 'wrapping paper', 'sketchbook', 'memo book', 'gift wrap set', 'sticker set'],
        'Gift Items': ['gift', 'voucher', 'cracker', 'tag', 'bunting', 'panettone', 'banner', 'party', 'cake case', 'decoupage', 'gift set', 'napkin', 'sticker', 'greeting', 'cardholder', 'gift bag', 'photo album', 'memory box', 'gift wrap', 'greeting card', 'passport cover', 'keepsake', 'gift box', 'photo book', 'trinket box', 'decorative tag'],
        'Lighting': ['light', 'lamp', 't-light', 'candle', 'lantern', 'led', 'bulb', 'night light', 'garland', 'fluted', 'hurricane lamp', 'candle plate', 'oil burner', 'candlestick', 'candle ring', 'bitty light', 'string lights', 'hanging light', 'table lamp', 'fairy lights'],
        'Storage': ['tin', 'box', 'drawer', 'organizer', 'cabinet', 'crate', 'basket', 'chest', 'cannister', 'jar', 'magazine rack', 'luggage tag', 'tidy', 'newspaper stand', 'tray oval', 'laundry box', 'tins', 'trinket pot', 'handy tin', 'sewing box', 'storage case', 'decorative cabinet', 'organizer tray', 'stacking tin'],
        'Bags': ['bag', 'backpack', 'rucksack', 'tote', 'handbag', 'charlotte', 'jumbo', 'shopper', 'washbag', 'lunch box', 'picnic', 'beach bag', 'shoulder bag', 'handy bucket', 'carry bag', 'tote bag', 'travel bag'],
        'Toys & Games': ['game', 'toy', 'puppet', 'lolly maker', 'playing card', 'mould', 'kit', 'jigsaw', 'dominoes', 'skittles', 'croquet', 'rounders', 'ludo', 'knitting', 'top trumps', 'soft toy', 'creature', 'helicopter', 'inflatable', 'racing car', 'craft kit', 'puzzle', 'doll', 'board game'],
        'Fashion Accessories': ['necklace', 'hairslide', 'jewel', 'trinket', 'earring', 'pendant', 'bead', 'charm', 'bracelet', 'brooch', 'hairband', 'hairclip', 'ring', 'hair tie', 'hair grip', 'lariat', 'bangle', 'choker', 'leis', 'hair accessory', 'necklace w tassel', 'raincoat', 'jewelry set', 'hair comb', 'expandable ring', 'hair slide'],
        'Garden & Outdoor': ['watering can', 'dovecote', 'garden', 'planter', 'bird feeder', 'bird table', 'kneeling pad', 'pot holder', 'thermometer', 'rake', 'spade', 'trowel', 'gloves', 'ladder', 'frisbee', 'grow', 'windchime', 'parasol', 'swing', 'birdhouse', 'gnome', 'plant cage', 'garden tool', 'birdhouse', 'outdoor decor', 'gardening set'],
        'Party Supplies': ['bunting', 'banner', 'cordon', 'party', 'balloon', 'confetti', 'invites', 'paper cup', 'party bag', 'chopstick', 'party decoration', 'cocktail accessory', 'paper plate'],
        'Personal Care': ['lip gloss', 'personal care', 'bath salts', 'cotton wool', 'cosy', 'soap dish', 'bathroom set', 'shower cap', 'balm', 'fragrance oil', 'bath accessory', 'scented oil', 'hot water bottle'],
        'Pet Supplies': ['cat', 'dog', 'dog bowl', 'cat bowl', 'dog collar', 'dog lead', 'canister', 'cat collar', 'pet toy', 'pet bowl', 'pet accessory'],
        'Furniture': ['stool', 'table', 'chair', 'bench', 'dresser', 'cabinet', 'folding stool', 'bar stool'],
        'Tech Accessories': ['usb lamp', 'headphones', 'electronic accessory', 'wireless device', 'torch'],
        'Other': []  # Fallback for uncategorized products
    }

    # Extract Product Category:
    def extract_category(description):
        if pd.isna(description):
            return 'Other'

        description = str(description).lower()

        # Check each category for keyword matches
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in description:
                    return category

        # If no matches found, return 'Other'
        return 'Other'
    data_df['ProductCategory'] = data_df['Description'].apply(extract_category)

    # Add a WeekStart column to track the date of the first day in the week:
    data_df['WeekStart'] = data_df['InvoiceDate'].dt.to_period('W').apply(lambda r: r.start_time)
    data_df['WeekStart'] = pd.to_datetime(data_df['WeekStart'])
    # Turn to a csv to be used by the model training function, overriding an older file if existed.
    print(data_df.shape)
    # Until HERE, EVERYTHING IS FINE
    weekly_data = data_df.groupby(['WeekStart', 'ProductCategory']).agg(
        Weekly_Revenue=('Revenue', 'sum'),
        Avg_UnitPrice=('UnitPrice', 'mean')  # Keeping price as a category characteristic
    ).reset_index()
    # Focus on time-based features
    weekly_data['Year'] = pd.to_datetime(weekly_data['WeekStart']).dt.year
    weekly_data['Month'] = pd.to_datetime(weekly_data['WeekStart']).dt.month
    weekly_data['WeekOfYear'] = pd.to_datetime(weekly_data['WeekStart']).dt.isocalendar().week

    # Add seasonal indicators
    weekly_data['IsSummer'] = ((weekly_data['Month'] >= 6) & (weekly_data['Month'] <= 8)).astype(int)
    weekly_data['IsHolidaySeason'] = ((weekly_data['Month'] >= 11) | (weekly_data['Month'] <= 1)).astype(int)
    weekly_data['IsSpring'] = ((weekly_data['Month'] >= 3) & (weekly_data['Month'] <= 5)).astype(int)
    weekly_data['IsFall'] = ((weekly_data['Month'] >= 9) & (weekly_data['Month'] <= 10)).astype(int)

    # Add lagged features (previous weeks' revenue)
    weekly_data = weekly_data.sort_values(['ProductCategory', 'WeekStart'])
    weekly_data['Prev_Week_Revenue'] = weekly_data.groupby('ProductCategory')['Weekly_Revenue'].shift(1)
    weekly_data['Prev_2_Week_Revenue'] = weekly_data.groupby('ProductCategory')['Weekly_Revenue'].shift(2)
    weekly_data['Prev_3_Week_Revenue'] = weekly_data.groupby('ProductCategory')['Weekly_Revenue'].shift(3)

    # Remove rows with NaN values (first weeks of each category where lag features are missing)
    weekly_data = weekly_data.dropna()

    # Cap values at 90th percentile => This capping is important to handle sneaky outliers
    # It increased performance dramatically!
    cap_value = weekly_data['Weekly_Revenue'].quantile(0.90)
    weekly_data['Weekly_Revenue'] = weekly_data['Weekly_Revenue'].clip(upper=cap_value)

    # Step 3: Prepare data for modeling
    # Define X (features)
    X = weekly_data[[
        # Category
        'ProductCategory',
        # Time features
        'Year',
        'Month',
        'WeekOfYear',
        # Seasonal indicators
        'IsSummer',
        'IsHolidaySeason',
        'IsSpring',
        'IsFall',
        # Historical patterns
        'Prev_Week_Revenue',
        'Prev_2_Week_Revenue',
        'Prev_3_Week_Revenue',
    ]]

    # Define y (target) - using log transform to handle outliers
    y = weekly_data['Weekly_Revenue']

    # Convert categorical variables
    # Identifying 'ProductCategory' as a categorical column
    categorical_cols = ['ProductCategory']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Use a column transformer for preprocessing
    # Using ColumnTransformer with OneHotEncoder to transform it properly
    # Applying StandardScaler to the numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # We'll use TimeSeriesSplit for proper time-based validation
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # Use the last fold

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Function to evaluate model performance
    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"{model_name} performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")

        return mae, rmse, r2

    
    # XGBoost
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    xgb_metrics = evaluate_model(y_test, y_pred_xgb, "XGBoost")
    print(xgb_metrics)
    
    # Define custom scoring metrics for evaluating models during tuning
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    scoring = {
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'rmse': make_scorer(rmse, greater_is_better=False),
        'r2': make_scorer(r2_score, greater_is_better=True)
    }
    # XGBOOST HYPERPARAMETER TUNING
    # Define parameter distributions for RandomizedSearchCV
    xgb_param_dist = {
        'regressor__n_estimators': randint(50, 300),
        'regressor__max_depth': randint(3, 10),
        'regressor__learning_rate': uniform(0.01, 0.3),
        'regressor__subsample': uniform(0.6, 0.4),
        'regressor__colsample_bytree': uniform(0.6, 0.4),
        'regressor__min_child_weight': randint(1, 10),
        'regressor__gamma': uniform(0, 1)
    }

    # Create RandomizedSearchCV for XGBoost
    xgb_random = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_param_dist,
        n_iter=25,  # Number of parameter settings sampled
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        scoring='neg_mean_absolute_error',
        random_state=42,
        return_train_score=True
    )

    # Fit RandomizedSearchCV
    print("\nStarting XGBoost RandomizedSearchCV...")
    xgb_random.fit(X, y)  # Using full dataset for tuning

    # Print best parameters and score
    print(f"Best XGBoost Parameters: {xgb_random.best_params_}")
    print(f"Best MAE Score: {-xgb_random.best_score_:.2f}")

    # Evaluate the best XGBoost model
    best_xgb = xgb_random.best_estimator_
    y_pred_best_xgb = best_xgb.predict(X_test)
    mae, rmse, r2 = evaluate_model(y_test, y_pred_best_xgb, "Tuned XGBoost")
    
    # Save metrics for the admin page to use
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'best_params': xgb_random.best_params_,
        'best_mae_score': -xgb_random.best_score_
    }
    
    # Save metrics to be used by the admin page
    joblib.dump(metrics, 'best_model_metrics.pkl')
    
    # dump the model to the specified path
    joblib.dump(best_xgb, old_model)
    
    print(weekly_data.shape)
    print(weekly_data.info())