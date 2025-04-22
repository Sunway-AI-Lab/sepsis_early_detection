
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tensorflow.keras import backend as K

# --- Configuration ---
FILE_PATH = 'sepsis-detection/data/raw/dataSepsis.csv'
TARGET_COLUMN = 'isSepsis'
SAVE_DIR = 'sepsis_model_files'
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'sepsis_model.keras')
IMPUTER_SAVE_PATH = os.path.join(SAVE_DIR, 'imputer.joblib')
SCALER_SAVE_PATH = os.path.join(SAVE_DIR, 'scaler.joblib')
FEATURES_SAVE_PATH = os.path.join(SAVE_DIR, 'feature_list.joblib')

# Model & Training Params
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
USE_SMOTE = True
K_FOLDS = 3  # Reduced for faster experimentation
MISSING_THRESHOLD = 0.9  # Drop features with >90% missing values
THRESHOLD = 0.3  # Adjusted threshold for higher recall

# --- 1. Load Data ---
print(f"Loading data from: {FILE_PATH}")
try:
    df = pd.read_csv(FILE_PATH, sep=';')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Feature Engineering & Selection ---
print(f"\nTarget variable ('{TARGET_COLUMN}') distribution:")
print(df[TARGET_COLUMN].value_counts(normalize=True))

if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found.")
    exit()

y = df[TARGET_COLUMN]
X = df.drop(TARGET_COLUMN, axis=1)

# Drop features with excessive missing values
missing_proportion = X.isnull().mean()
features_to_drop = missing_proportion[missing_proportion > MISSING_THRESHOLD].index.tolist()
print(f"\nDropping {len(features_to_drop)} features with >{MISSING_THRESHOLD*100}% missing values: {features_to_drop}")
X = X.drop(features_to_drop, axis=1)

# Clinically relevant features (based on sepsis literature)
priority_features = ['HR', 'Temp', 'Resp', 'WBC', 'Lactate', 'MAP', 'Creatinine', 'BUN', 'Glucose', 'Potassium', 'Age', 'ICULOS']
# Ensure only available priority features are used
priority_features = [f for f in priority_features if f in X.columns]

# Feature importance using RandomForest
print("\nCalculating feature importance...")
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
rf.fit(X.fillna(X.median()), y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 important features:")
print(feature_importance.head(10))

# Select top features (combine priority features and top N from RF)
N_TOP_FEATURES = 15
top_features = list(set(priority_features + feature_importance.head(N_TOP_FEATURES).index.tolist()))
X = X[top_features]
features_list = X.columns.tolist()
print(f"\nSelected {len(features_list)} features: {features_list}")

# --- 3. Data Splitting ---
print("\n--- Splitting Data ---")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

val_split_ratio = VALIDATION_SIZE / (1.0 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_split_ratio, random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Training set shape:   {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape:       {X_test.shape}")

# --- 4. Data Preprocessing ---
print("\n--- Preprocessing Data ---")

# Imputation with KNN
print("Applying KNN Imputation...")
imputer = KNNImputer(n_neighbors=5)
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Scaling with RobustScaler
print("Applying RobustScaler...")
scaler = RobustScaler()
scaler.fit(X_train_imputed)
X_train_scaled = scaler.transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features_list, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=features_list, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features_list, index=X_test.index)
print("Preprocessing complete. NaNs remaining:")
print(f"Train: {X_train_scaled.isnull().sum().sum()}")
print(f"Val: {X_val_scaled.isnull().sum().sum()}")
print(f"Test: {X_test_scaled.isnull().sum().sum()}")

# Apply SMOTE
if USE_SMOTE:
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)  # Partial oversampling
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE, training set shape: {X_train_scaled.shape}")
    print(f"New training target distribution:\n{y_train.value_counts(normalize=True)}")

# --- 5. Focal Loss for Imbalanced Data ---
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        fl = tf.where(tf.equal(y_true, 1), fl, (1 - alpha) * tf.pow(pt, gamma) * tf.math.log(1 - pt))
        return tf.reduce_mean(fl)
    return focal_loss_fn

# --- 6. Build the Model ---
print("\n--- Building TensorFlow Model ---")
n_features = X_train_scaled.shape[1]
tf.random.set_seed(RANDOM_STATE)

def create_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), name='Input_Layer'),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

# --- 7. Cross-Validation ---
print("\n--- Performing Cross-Validation ---")
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"\nFold {fold + 1}/{K_FOLDS}")
    X_fold_train, X_fold_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = create_model(n_features)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1
    )
    
    history = model.fit(
        X_fold_train, y_fold_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_fold_val, y_fold_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    val_auc = max(history.history['val_auc'])
    cv_scores.append(val_auc)
    print(f"Fold {fold + 1} AUC: {val_auc:.4f}")

print(f"\nMean CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# --- 8. Train Final Model ---
print("\n--- Training Final Model ---")
model = create_model(n_features)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# --- 9. Evaluate Model ---
print("\n--- Evaluating Model on Test Set ---")
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)})
loss, accuracy, auc_score, precision, recall = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Set Evaluation:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC: {auc_score:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")

y_pred_proba = best_model.predict(X_test_scaled).flatten()
y_pred_class = (y_pred_proba > THRESHOLD).astype(int)  # Custom threshold

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=['No Sepsis (0)', 'Sepsis (1)'], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Precision-Recall Curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)
plt.figure(figsize=(7, 6))
plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# --- 10. Plot Training History ---
def plot_training_history(history):
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i + 1)
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# --- 11. Save Preprocessing Objects ---
print("\n--- Saving Preprocessing Objects ---")
joblib.dump(imputer, IMPUTER_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
joblib.dump(features_list, FEATURES_SAVE_PATH)
print(f"Saved imputer, scaler, and feature list to {SAVE_DIR}")

# --- 12. Prediction Function ---
def predict_sepsis(new_data, model, imputer, scaler, feature_list):
    try:
        input_df = pd.DataFrame(new_data, columns=feature_list)
        for col in feature_list:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[feature_list]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        probabilities = model.predict(input_scaled).flatten()
        classes = (probabilities > THRESHOLD).astype(int)
        return probabilities, classes
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# --- Example Prediction ---
print("\n--- Example Prediction ---")
sample_1 = {col: X_train[col].median() for col in features_list}
sample_1['HR'] = 120
sample_1['Temp'] = 38.8
sample_1['Resp'] = 28
sample_1['WBC'] = 15.5
sample_1['Lactate'] = 2.9
sample_1['BUN'] = 35

sample_2 = {col: X_train[col].median() for col in features_list}
sample_2['HR'] = 75
sample_2['Temp'] = 36.6
sample_2['Resp'] = 16
sample_2['WBC'] = 8.0
sample_2['Lactate'] = 1.1
sample_2['BUN'] = 15

samples = [sample_1, sample_2]
probs, classes = predict_sepsis(samples, best_model, imputer, scaler, features_list)

if probs is not None:
    print("\nPrediction Results:")
    for i, (prob, cls) in enumerate(zip(probs, classes)):
        print(f"Sample {i+1}: Probability={prob:.4f}, Class={cls} ({'Sepsis' if cls == 1 else 'No Sepsis'})")

print("\n--- END OF SCRIPT ---")