# gpu_homework.py
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 60)
print("深度神經網絡作業")
print("=" * 60)

# 設置 GPU 配置
def setup_gpu():
    # 檢查可用 GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"🎯 檢測到 {len(gpus)} 個 GPU:")
        for gpu in gpus:
            print(f"   - {gpu}")
        
        try:
            # 設置 GPU 內存增長
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 設置邏輯 GPU
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"✅ 邏輯 GPU: {len(logical_gpus)}")
            
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU 設置警告: {e}")
            return False
    else:
        print("❌ 未檢測到 GPU，將使用 CPU")
        return False

# 設置 GPU
gpu_available = setup_gpu()

# 加載 CIFAR10 數據集
print("\n📦 加載 CIFAR10 數據集...")
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# 數據預處理
print("🔄 數據預處理...")
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 展平圖像 (32x32x3 = 3072)
X_train_full = X_train_full.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)

# 分割訓練集和驗證集
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

print(f"📊 數據集信息:")
print(f"   訓練集: {X_train.shape[0]} 個樣本")
print(f"   驗證集: {X_valid.shape[0]} 個樣本")
print(f"   測試集: {X_test.shape[0]} 個樣本")
print(f"   特徵維度: {X_train.shape[1]}")

# 1. 基線模型 (20層 DNN，He初始化，ELU激活)
def build_baseline_model():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    
    # 20個隱藏層，每層100個神經元
    for i in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# 2. Batch Normalization 模型
def build_batchnorm_model():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    
    for i in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# 3. SELU 自歸一化網絡
def build_selu_model():
    # 標準化輸入特徵 (SELU 的要求)
    scaler = StandardScaler()
    X_train_selu = scaler.fit_transform(X_train)
    X_valid_selu = scaler.transform(X_valid)
    X_test_selu = scaler.transform(X_test)
    
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    
    for i in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
        model.add(keras.layers.Activation("selu"))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model, X_train_selu, X_valid_selu, X_test_selu, scaler

# 4. Alpha Dropout 模型
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
        if i < 19:  # 不在最後一層前添加 dropout
            model.add(keras.layers.AlphaDropout(rate=0.1))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model, X_train_alpha, X_valid_alpha, X_test_alpha, scaler

# 學習率查找函數
def find_learning_rate(model, X, y, validation_data, start_lr=1e-6, end_lr=1e-1, epochs=5):
    print("   🔍 尋找最佳學習率...")
    
    # 測試幾個學習率
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
        
        # 快速訓練
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
    
    print(f"   ✅ 最佳學習率: {best_lr}")
    return best_lr

# 訓練函數
def train_model(model, X_train, y_train, X_valid, y_valid, model_name, lr=0.001, epochs=100):
    print(f"\n🚀 訓練 {model_name}...")
    
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 早停法
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # 使用 GPU 友好的 batch size
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

# MC Dropout 預測
def mc_dropout_predict(model, X, n_samples=50):
    print(f"   🔄 MC Dropout 預測 ({n_samples} 次採樣)...")
    y_probas = []
    for i in range(n_samples):
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
    print(f"   📊 {model_name}{mc_text} 測試準確率: {accuracy:.4f}")
    return accuracy

# 主訓練流程
print("\n" + "="*50)
print("開始模型訓練")
print("="*50)

# 1. 基線模型
print("\n1️⃣ 基線模型 (20層 DNN + ELU + He初始化)")
baseline_model = build_baseline_model()
baseline_lr = find_learning_rate(baseline_model, X_train, y_train, (X_valid, y_valid))
baseline_history = train_model(baseline_model, X_train, y_train, X_valid, y_valid, 
                              "基線模型", baseline_lr, 100)

# 2. BatchNorm 模型
print("\n2️⃣ Batch Normalization 模型")
batchnorm_model = build_batchnorm_model()
batchnorm_lr = find_learning_rate(batchnorm_model, X_train, y_train, (X_valid, y_valid))
batchnorm_history = train_model(batchnorm_model, X_train, y_train, X_valid, y_valid,
                               "BatchNorm模型", batchnorm_lr, 100)

# 3. SELU 模型
print("\n3️⃣ SELU 自歸一化模型")
selu_model, X_train_selu, X_valid_selu, X_test_selu, selu_scaler = build_selu_model()
selu_lr = find_learning_rate(selu_model, X_train_selu, y_train, (X_valid_selu, y_valid))
selu_history = train_model(selu_model, X_train_selu, y_train, X_valid_selu, y_valid,
                          "SELU模型", selu_lr, 100)

# 4. Alpha Dropout 模型
print("\n4️⃣ Alpha Dropout 模型")
alpha_model, X_train_alpha, X_valid_alpha, X_test_alpha, alpha_scaler = build_alpha_dropout_model()
alpha_lr = find_learning_rate(alpha_model, X_train_alpha, y_train, (X_valid_alpha, y_valid))
alpha_history = train_model(alpha_model, X_train_alpha, y_train, X_valid_alpha, y_valid,
                           "AlphaDropout模型", alpha_lr, 100)

# 評估所有模型
print("\n" + "="*50)
print("模型評估")
print("="*50)

baseline_acc = evaluate_model(baseline_model, X_test, y_test, "基線模型")
batchnorm_acc = evaluate_model(batchnorm_model, X_test, y_test, "BatchNorm模型")
selu_acc = evaluate_model(selu_model, X_test_selu, y_test, "SELU模型")
alpha_acc = evaluate_model(alpha_model, X_test_alpha, y_test, "AlphaDropout模型")
alpha_mc_acc = evaluate_model(alpha_model, X_test_alpha, y_test, "AlphaDropout模型", use_mc_dropout=True)

# 結果分析
print("\n" + "="*50)
print("結果總結")
print("="*50)

results = [
    ("基線模型", baseline_acc),
    ("BatchNorm模型", batchnorm_acc),
    ("SELU模型", selu_acc),
    ("AlphaDropout模型", alpha_acc),
    ("AlphaDropout+MC", alpha_mc_acc)
]

print("📈 測試準確率:")
for name, acc in results:
    improvement = acc - baseline_acc
    print(f"   {name:20} {acc:.4f} ({improvement:+.4f})")

# 繪製學習曲線
print("\n📊 繪製學習曲線...")
plt.figure(figsize=(16, 12))

# 訓練損失
plt.subplot(2, 2, 1)
plt.plot(baseline_history.history['loss'], label='基線模型', linewidth=2)
plt.plot(batchnorm_history.history['loss'], label='BatchNorm模型', linewidth=2)
plt.plot(selu_history.history['loss'], label='SELU模型', linewidth=2)
plt.plot(alpha_history.history['loss'], label='AlphaDropout模型', linewidth=2)
plt.title('訓練損失', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 驗證準確率
plt.subplot(2, 2, 2)
plt.plot(baseline_history.history['val_accuracy'], label='基線模型', linewidth=2)
plt.plot(batchnorm_history.history['val_accuracy'], label='BatchNorm模型', linewidth=2)
plt.plot(selu_history.history['val_accuracy'], label='SELU模型', linewidth=2)
plt.plot(alpha_history.history['val_accuracy'], label='AlphaDropout模型', linewidth=2)
plt.title('驗證準確率', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 測試準確率比較
plt.subplot(2, 2, 3)
model_names = [r[0] for r in results]
accuracies = [r[1] for r in results]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

bars = plt.bar(model_names, accuracies, color=colors, alpha=0.7)
plt.title('測試準確率比較', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)

# 添加數值標籤
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# 收斂速度分析
plt.subplot(2, 2, 4)
convergence_data = [
    (len(baseline_history.history['loss']), '基線模型'),
    (len(batchnorm_history.history['loss']), 'BatchNorm模型'),
    (len(selu_history.history['loss']), 'SELU模型'),
    (len(alpha_history.history['loss']), 'AlphaDropout模型')
]

epochs = [d[0] for d in convergence_data]
names = [d[1] for d in convergence_data]

plt.bar(names, epochs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
plt.title('收斂速度 (訓練輪數)', fontsize=14, fontweight='bold')
plt.ylabel('Epochs')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)

for i, (epoch, name) in enumerate(zip(epochs, names)):
    plt.text(i, epoch + 0.5, str(epoch), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 技術分析
print("\n" + "="*50)
print("技術分析")
print("="*50)

print("1. 📊 Batch Normalization 效果:")
print(f"   - 改進: {batchnorm_acc - baseline_acc:+.4f}")
print("   - 觀察: 應該有更快的收斂和更好的泛化")

print("\n2. 🔄 SELU 自歸一化效果:")
print(f"   - 改進: {selu_acc - baseline_acc:+.4f}")
print("   - 觀察: 無需顯式歸一化層的自歸一化特性")

print("\n3. 🛡️ Alpha Dropout 效果:")
print(f"   - 基礎改進: {alpha_acc - baseline_acc:+.4f}")
print(f"   - MC Dropout 額外改進: {alpha_mc_acc - alpha_acc:+.4f}")
print("   - 觀察: 為 SELU 網絡設計的正則化")

print("\n4. ⚡ 訓練速度:")
print(f"   - 基線模型: {len(baseline_history.history['loss'])} 輪")
print(f"   - BatchNorm: {len(batchnorm_history.history['loss'])} 輪")
print(f"   - SELU: {len(selu_history.history['loss'])} 輪")
print(f"   - AlphaDropout: {len(alpha_history.history['loss'])} 輪")

print(f"\n5. 🖥️ 硬件使用:")
print(f"   - GPU 加速: {'是' if gpu_available else '否'}")
if gpu_available:
    print(f"   - Batch Size: 512")
else:
    print(f"   - Batch Size: 128")

print("\n🎉 作業完成！")
