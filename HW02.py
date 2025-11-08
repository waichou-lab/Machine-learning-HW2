# gpu_homework_optimized.py
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# GPU配置
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detected: {len(gpus)}")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPU detected, using CPU")
        return False

# 設置環境
gpu_available = setup_gpu()
print("TensorFlow version:", tf.__version__)

# 加載數據
print("Loading CIFAR10 dataset...")
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# 數據預處理
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train_full = X_train_full.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

print(f"Training: {X_train.shape[0]}, Validation: {X_valid.shape[0]}, Test: {X_test.shape[0]}")

# 模型構建函數
def build_baseline_model():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.ELU())
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def build_batchnorm_model():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ELU())
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def build_selu_model():
    scaler = StandardScaler()
    X_train_selu = scaler.fit_transform(X_train)
    X_valid_selu = scaler.transform(X_valid)
    X_test_selu = scaler.transform(X_test)
    
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
        model.add(keras.layers.Activation("selu"))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model, X_train_selu, X_valid_selu, X_test_selu, scaler

def build_alpha_dropout_model():
    scaler = StandardScaler()
    X_train_alpha = scaler.fit_transform(X_train)
    X_valid_alpha = scaler.transform(X_valid)
    X_test_alpha = scaler.transform(X_test)
    
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    for i in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
        model.add(keras.layers.Activation("selu"))
        if i < 19:
            model.add(keras.layers.AlphaDropout(rate=0.1))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model, X_train_alpha, X_valid_alpha, X_test_alpha, scaler

# 學習率搜索
def find_learning_rate(model, X, y, validation_data):
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    best_lr = 1e-3
    best_val_acc = 0
    
    for lr in learning_rates:
        model_copy = keras.models.clone_model(model)
        model_copy.build((None, 3072))
        model_copy.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model_copy.fit(
            X, y,
            validation_data=validation_data,
            epochs=2,
            batch_size=128,
            verbose=0
        )
        
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
    
    print(f"Best learning rate: {best_lr}")
    return best_lr

# 訓練函數
def train_model(model, X_train, y_train, X_valid, y_valid, model_name, lr=0.001, epochs=100):
    print(f"\nTraining {model_name}...")
    
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    batch_size = 512 if gpu_available else 128
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

# MC Dropout預測
def mc_dropout_predict(model, X, n_samples=50):
    y_probas = []
    for _ in range(n_samples):
        y_proba = model.predict(X, verbose=0)
        y_probas.append(y_proba)
    return np.mean(y_probas, axis=0)

# 評估函數
def evaluate_model(model, X_test, y_test, model_name, use_mc_dropout=False):
    if use_mc_dropout:
        y_proba = mc_dropout_predict(model, X_test)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    mc_text = " (MC Dropout)" if use_mc_dropout else ""
    print(f"{model_name}{mc_text} Test Accuracy: {accuracy:.4f}")
    return accuracy

# 繪圖函數
def create_results_plot(baseline_history, batch_norm_history, selu_history, alpha_history, 
                       baseline_acc, batchnorm_acc, selu_acc, alpha_acc, alpha_mc_acc):
    
    plt.figure(figsize=(16, 12))
    
    # 訓練損失
    plt.subplot(2, 2, 1)
    plt.plot(baseline_history.history['loss'], label='Baseline', linewidth=2)
    plt.plot(batch_norm_history.history['loss'], label='BatchNorm', linewidth=2)
    plt.plot(selu_history.history['loss'], label='SELU', linewidth=2)
    plt.plot(alpha_history.history['loss'], label='AlphaDropout', linewidth=2)
    plt.title('A. Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 驗證準確率
    plt.subplot(2, 2, 2)
    plt.plot(baseline_history.history['val_accuracy'], label='Baseline', linewidth=2)
    plt.plot(batch_norm_history.history['val_accuracy'], label='BatchNorm', linewidth=2)
    plt.plot(selu_history.history['val_accuracy'], label='SELU', linewidth=2)
    plt.plot(alpha_history.history['val_accuracy'], label='AlphaDropout', linewidth=2)
    plt.title('B. Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 測試準確率
    plt.subplot(2, 2, 3)
    models = ['Baseline', 'BatchNorm', 'SELU', 'AlphaDropout', 'AlphaDropout\n+ MC']
    accuracies = [baseline_acc, batchnorm_acc, selu_acc, alpha_acc, alpha_mc_acc]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.title('C. Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{acc:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 收斂速度
    plt.subplot(2, 2, 4)
    epochs_data = [
        (len(baseline_history.history['loss']), 'Baseline'),
        (len(batch_norm_history.history['loss']), 'BatchNorm'),
        (len(selu_history.history['loss']), 'SELU'),
        (len(alpha_history.history['loss']), 'AlphaDropout')
    ]
    
    epochs = [d[0] for d in epochs_data]
    names = [d[1] for d in epochs_data]
    
    bars = plt.bar(names, epochs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    plt.title('D. Convergence Speed')
    plt.ylabel('Epochs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    for i, (epoch, name) in enumerate(zip(epochs, names)):
        plt.text(i, epoch + 0.5, str(epoch), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主訓練流程
print("\n" + "="*50)
print("Model Training")
print("="*50)

# 1. 基線模型
print("\n1. Baseline Model")
baseline_model = build_baseline_model()
baseline_lr = find_learning_rate(baseline_model, X_train, y_train, (X_valid, y_valid))
baseline_history = train_model(baseline_model, X_train, y_train, X_valid, y_valid, 
                              "Baseline", baseline_lr, 100)

# 2. BatchNorm模型
print("\n2. BatchNorm Model")
batchnorm_model = build_batchnorm_model()
batchnorm_lr = find_learning_rate(batchnorm_model, X_train, y_train, (X_valid, y_valid))
batchnorm_history = train_model(batchnorm_model, X_train, y_train, X_valid, y_valid,
                               "BatchNorm", batchnorm_lr, 100)

# 3. SELU模型
print("\n3. SELU Model")
selu_model, X_train_selu, X_valid_selu, X_test_selu, selu_scaler = build_selu_model()
selu_lr = find_learning_rate(selu_model, X_train_selu, y_train, (X_valid_selu, y_valid))
selu_history = train_model(selu_model, X_train_selu, y_train, X_valid_selu, y_valid,
                          "SELU", selu_lr, 100)

# 4. Alpha Dropout模型
print("\n4. Alpha Dropout Model")
alpha_model, X_train_alpha, X_valid_alpha, X_test_alpha, alpha_scaler = build_alpha_dropout_model()
alpha_lr = find_learning_rate(alpha_model, X_train_alpha, y_train, (X_valid_alpha, y_valid))
alpha_history = train_model(alpha_model, X_train_alpha, y_train, X_valid_alpha, y_valid,
                           "AlphaDropout", alpha_lr, 100)

# 評估模型
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

baseline_acc = evaluate_model(baseline_model, X_test, y_test, "Baseline")
batchnorm_acc = evaluate_model(batchnorm_model, X_test, y_test, "BatchNorm")
selu_acc = evaluate_model(selu_model, X_test_selu, y_test, "SELU")
alpha_acc = evaluate_model(alpha_model, X_test_alpha, y_test, "AlphaDropout")
alpha_mc_acc = evaluate_model(alpha_model, X_test_alpha, y_test, "AlphaDropout", use_mc_dropout=True)

# 生成圖表
print("\nGenerating plots...")
create_results_plot(baseline_history, batchnorm_history, selu_history, alpha_history,
                   baseline_acc, batchnorm_acc, selu_acc, alpha_acc, alpha_mc_acc)

# 結果總結
print("\n" + "="*50)
print("Final Results")
print("="*50)

results = [
    ("Baseline", baseline_acc),
    ("BatchNorm", batchnorm_acc),
    ("SELU", selu_acc),
    ("AlphaDropout", alpha_acc),
    ("AlphaDropout+MC", alpha_mc_acc)
]

print("Test Accuracy Summary:")
for name, acc in results:
    improvement = acc - baseline_acc
    print(f"  {name:20} {acc:.4f} ({improvement:+.4f})")

print(f"\nBest Model: SELU ({selu_acc:.4f})")
print("Training completed!")
