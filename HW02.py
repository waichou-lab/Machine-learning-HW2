import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 设置随机种子以确保结果可重现
tf.random.set_seed(42)
np.random.seed(42)

# 加载CIFAR10数据集
print("Loading CIFAR10 dataset...")
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# 数据预处理
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 将图像展平为向量（32x32x3 = 3072维）
X_train_full = X_train_full.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)

# 分割训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_valid.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 定义构建模型的各种函数
def build_baseline_model(input_shape=3072, n_classes=10):
    """构建基线DNN模型（20个隐藏层，每层100个神经元）"""
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
    
    # 添加20个隐藏层
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    return model

def build_batch_norm_model(input_shape=3072, n_classes=10):
    """构建带Batch Normalization的DNN模型"""
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
    
    # 添加20个带BatchNorm的隐藏层
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    return model

def build_selu_model(input_shape=3072, n_classes=10):
    """构建使用SELU激活函数的自归一化网络"""
    # 标准化输入特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
    
    # 添加20个使用SELU的隐藏层（使用LeCun初始化）
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
        model.add(keras.layers.Activation("selu"))
    
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    
    return model, X_train_scaled, X_valid_scaled, X_test_scaled, scaler

def build_alpha_dropout_model(input_shape=3072, n_classes=10):
    """构建带Alpha Dropout的SELU模型"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
    
    # 添加20个使用SELU和Alpha Dropout的隐藏层
    for i in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
        model.add(keras.layers.Activation("selu"))
        if i < 19:  # 不在最后一层之前添加dropout
            model.add(keras.layers.AlphaDropout(rate=0.1))
    
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    
    return model, X_train_scaled, X_valid_scaled, X_test_scaled, scaler

# 学习率查找函数
def find_learning_rate(model, X, y, validation_data, start_lr=1e-6, end_lr=1.0, epochs=5):
    """使用学习率范围测试找到最佳学习率"""
    # 使用学习率调度器
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: start_lr * 10**(epoch / epochs * np.log10(end_lr/start_lr))
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Nadam(learning_rate=start_lr),
        metrics=["accuracy"]
    )
    
    history = model.fit(
        X, y,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=128,
        callbacks=[lr_schedule],
        verbose=0
    )
    
    # 找到损失下降最快的学习率
    losses = history.history['loss']
    lrs = [start_lr * 10**(epoch / epochs * np.log10(end_lr/start_lr)) for epoch in range(epochs)]
    
    # 计算损失的梯度
    grad_loss = np.gradient(losses)
    best_idx = np.argmin(grad_loss)
    best_lr = lrs[best_idx]
    
    return best_lr, lrs, losses

# 训练函数
def train_model(model, X_train, y_train, X_valid, y_valid, lr=0.001, epochs=100, model_name="model"):
    """训练模型并返回历史记录"""
    print(f"\n=== Training {model_name} ===")
    
    # 编译模型
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Nadam(learning_rate=lr),
        metrics=["accuracy"]
    )
    
    # 早停回调
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

# MC Dropout预测函数
def mc_dropout_predict(model, X, n_samples=100):
    """使用MC Dropout进行预测"""
    # 启用训练时的dropout层进行预测
    y_probas = []
    for _ in range(n_samples):
        y_proba = model.predict(X, verbose=0)
        y_probas.append(y_proba)
    
    return np.mean(y_probas, axis=0)

# 1. 基线模型（无BatchNorm）
print("\n" + "="*50)
print("1. TRAINING BASELINE MODEL")
print("="*50)

baseline_model = build_baseline_model()
baseline_lr, lrs, losses = find_learning_rate(
    build_baseline_model(), X_train, y_train, (X_valid, y_valid)
)
print(f"Found learning rate for baseline: {baseline_lr:.6f}")

baseline_history = train_model(
    baseline_model, X_train, y_train, X_valid, y_valid, 
    lr=baseline_lr, model_name="Baseline"
)

# 2. Batch Normalization模型
print("\n" + "="*50)
print("2. TRAINING BATCH NORM MODEL")
print("="*50)

batch_norm_model = build_batch_norm_model()
batch_norm_lr, _, _ = find_learning_rate(
    build_batch_norm_model(), X_train, y_train, (X_valid, y_valid)
)
print(f"Found learning rate for BatchNorm: {batch_norm_lr:.6f}")

batch_norm_history = train_model(
    batch_norm_model, X_train, y_train, X_valid, y_valid,
    lr=batch_norm_lr, model_name="BatchNorm"
)

# 3. SELU模型
print("\n" + "="*50)
print("3. TRAINING SELU MODEL")
print("="*50)

selu_model, X_train_selu, X_valid_selu, X_test_selu, selu_scaler = build_selu_model()
selu_lr, _, _ = find_learning_rate(
    build_selu_model()[0], X_train_selu, y_train, (X_valid_selu, y_valid)
)
print(f"Found learning rate for SELU: {selu_lr:.6f}")

selu_history = train_model(
    selu_model, X_train_selu, y_train, X_valid_selu, y_valid,
    lr=selu_lr, model_name="SELU"
)

# 4. Alpha Dropout模型
print("\n" + "="*50)
print("4. TRAINING ALPHA DROPOUT MODEL")
print("="*50)

alpha_dropout_model, X_train_alpha, X_valid_alpha, X_test_alpha, alpha_scaler = build_alpha_dropout_model()
alpha_lr, _, _ = find_learning_rate(
    build_alpha_dropout_model()[0], X_train_alpha, y_train, (X_valid_alpha, y_valid)
)
print(f"Found learning rate for Alpha Dropout: {alpha_lr:.6f}")

alpha_history = train_model(
    alpha_dropout_model, X_train_alpha, y_train, X_valid_alpha, y_valid,
    lr=alpha_lr, model_name="Alpha Dropout"
)

# 5. 使用Icycle调度的模型
print("\n" + "="*50)
print("5. TRAINING WITH ICYCLE SCHEDULING")
print("="*50)

def build_icycle_model():
    """构建用于Icycle调度的模型"""
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3072,)))
    
    for _ in range(20):
        model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ELU())
    
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model

icycle_model = build_icycle_model()

# 定义Icycle调度
class IcycleScheduler(keras.callbacks.Callback):
    def __init__(self, max_lr, min_lr, step_size):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size = step_size
        self.iterations = 0
    
    def on_batch_begin(self, batch, logs=None):
        # 三角循环学习率
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
        lr = self.min_lr + (self.max_lr - self.min_lr) * max(0, 1 - x)
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.iterations += 1

# 使用Icycle调度训练
icycle_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    metrics=["accuracy"]
)

icycle_scheduler = IcycleScheduler(max_lr=0.01, min_lr=0.0001, step_size=500)

early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

icycle_history = icycle_model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=128,
    callbacks=[icycle_scheduler, early_stopping],
    verbose=1
)

# 评估所有模型
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

def evaluate_model(model, X_test, y_test, model_name, use_mc_dropout=False):
    """评估模型性能"""
    if use_mc_dropout:
        y_proba = mc_dropout_predict(model, X_test)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")
    return accuracy

# 评估基线模型
baseline_acc = evaluate_model(baseline_model, X_test, y_test, "Baseline")

# 评估BatchNorm模型
batchnorm_acc = evaluate_model(batch_norm_model, X_test, y_test, "BatchNorm")

# 评估SELU模型
selu_acc = evaluate_model(selu_model, X_test_selu, y_test, "SELU")

# 评估Alpha Dropout模型（常规预测）
alpha_acc = evaluate_model(alpha_dropout_model, X_test_alpha, y_test, "Alpha Dropout")

# 评估Alpha Dropout模型（MC Dropout）
alpha_mc_acc = evaluate_model(alpha_dropout_model, X_test_alpha, y_test, "Alpha Dropout (MC)", use_mc_dropout=True)

# 评估Icycle模型
icycle_acc = evaluate_model(icycle_model, X_test, y_test, "Icycle")

# 绘制学习曲线
plt.figure(figsize=(15, 10))

# 训练损失
plt.subplot(2, 2, 1)
plt.plot(baseline_history.history['loss'], label='Baseline')
plt.plot(batch_norm_history.history['loss'], label='BatchNorm')
plt.plot(selu_history.history['loss'], label='SELU')
plt.plot(alpha_history.history['loss'], label='Alpha Dropout')
plt.plot(icycle_history.history['loss'], label='Icycle')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 验证准确率
plt.subplot(2, 2, 2)
plt.plot(baseline_history.history['val_accuracy'], label='Baseline')
plt.plot(batch_norm_history.history['val_accuracy'], label='BatchNorm')
plt.plot(selu_history.history['val_accuracy'], label='SELU')
plt.plot(alpha_history.history['val_accuracy'], label='Alpha Dropout')
plt.plot(icycle_history.history['val_accuracy'], label='Icycle')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 测试准确率比较
plt.subplot(2, 2, 3)
models = ['Baseline', 'BatchNorm', 'SELU', 'Alpha\nDropout', 'Alpha\nDropout(MC)', 'Icycle']
accuracies = [baseline_acc, batchnorm_acc, selu_acc, alpha_acc, alpha_mc_acc, icycle_acc]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

# 在柱状图上添加数值标签
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 结果总结
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"Baseline Model Test Accuracy: {baseline_acc:.4f}")
print(f"BatchNorm Model Test Accuracy: {batchnorm_acc:.4f}")
print(f"SELU Model Test Accuracy: {selu_acc:.4f}")
print(f"Alpha Dropout Model Test Accuracy: {alpha_acc:.4f}")
print(f"Alpha Dropout with MC Dropout Test Accuracy: {alpha_mc_acc:.4f}")
print(f"Icycle Model Test Accuracy: {icycle_acc:.4f}")

# 分析每个技术的效果
print("\n" + "="*50)
print("TECHNIQUE ANALYSIS")
print("="*50)
print("1. Batch Normalization:")
print(f"   - Improvement over baseline: {batchnorm_acc - baseline_acc:+.4f}")
print("   - Expected: Faster convergence and better generalization")

print("\n2. SELU vs BatchNorm:")
print(f"   - SELU improvement over BatchNorm: {selu_acc - batchnorm_acc:+.4f}")
print("   - Expected: Self-normalizing properties without explicit normalization")

print("\n3. Alpha Dropout:")
print(f"   - Improvement over SELU: {alpha_acc - selu_acc:+.4f}")
print("   - MC Dropout improvement: {alpha_mc_acc - alpha_acc:+.4f}")
print("   - Expected: Better regularization for SELU networks")

print("\n4. Icycle Scheduling:")
print(f"   - Improvement over BatchNorm: {icycle_acc - batchnorm_acc:+.4f}")
print("   - Expected: Faster training and better convergence")