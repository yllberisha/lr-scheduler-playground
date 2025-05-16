import streamlit as st
import math
import random
import numpy as np
import plotly.graph_objs as go
import inspect
import tempfile
import os
# ---------------- Python implementations ----------------

def cosine_decay(initial_lr, step, decay_steps):
    if step > decay_steps:
        return 0.0
    return initial_lr * 0.5 * (1 + math.cos(math.pi * step / decay_steps))


def linear_cosine_decay(initial_lr, step, decay_steps, alpha=0.0, beta=0.0):
    if step > decay_steps:
        return 0.0
    t = step / decay_steps
    cosine_part = 0.5 * (1 + math.cos(math.pi * t))
    return initial_lr * (alpha + (1 - alpha) * cosine_part + beta * t)


def noisy_linear_cosine_decay(initial_lr, step, decay_steps, alpha=0.0, beta=0.0, noise_std=1.0):
    base = linear_cosine_decay(initial_lr, step, decay_steps, alpha, beta)
    return max(0.0, base + random.gauss(0, noise_std))


def cyclical_lr(step, step_size, base_lr, max_lr):
    cycle = math.floor(1 + step / (2 * step_size))
    x = abs(step / step_size - 2 * cycle + 1)
    scale = max(0, 1 - x)
    return base_lr + (max_lr - base_lr) * scale


def exp_cyclical_lr(step, step_size, base_lr, max_lr, gamma=0.9999):
    return cyclical_lr(step, step_size, base_lr, max_lr) * (gamma ** step)


def custom_schedule(step):
    return 1.0 / math.sqrt(step + 1)


# ---------------- Scheduler registry & defaults ----------------

def get_schedulers_and_defaults():
    schedulers = {
        'Cosine Decay': cosine_decay,
        'Linear Cosine Decay': linear_cosine_decay,
        'Noisy Linear Cosine Decay': noisy_linear_cosine_decay,
        'Cyclical Learning Rate': cyclical_lr,
        'Exponential Cyclical Learning Rate': exp_cyclical_lr,
        'Custom Learning Scheduler': (lambda step, **kw: custom_schedule(step))
    }
    defaults = {
        'Cosine Decay':                     {'initial_lr': 0.001, 'decay_steps': 100},
        'Linear Cosine Decay':              {'initial_lr': 0.001, 'decay_steps': 100, 'alpha': 0.0, 'beta': 0.0},
        'Noisy Linear Cosine Decay':        {'initial_lr': 0.001, 'decay_steps': 100, 'alpha': 0.0, 'beta': 0.0, 'noise_std': 0.1},
        'Cyclical Learning Rate':           {'step_size': 50,  'base_lr': 0.0005, 'max_lr': 0.003},
        'Exponential Cyclical Learning Rate': {'step_size': 50, 'base_lr': 0.0005, 'max_lr': 0.003, 'gamma': 0.9999},
        'Custom Learning Scheduler':        {}
    }
    return schedulers, defaults

schedulers, defaults = get_schedulers_and_defaults()

# ---------------- Streamlit UI ----------------

st.set_page_config(layout="wide", page_title="LR Scheduler Playground", page_icon="⚙️")
st.title("🔬 Learning Rate Scheduler Playground")

# Global settings
st.sidebar.header("Global Settings")
force_cpu = st.sidebar.checkbox("Force CPU (disable GPU)", value=True)
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # must precede TF import

# Scheduler parameters
st.sidebar.header("Scheduler Settings")
choice = st.sidebar.selectbox("Select Scheduler", list(schedulers.keys()))
params = {}
for p, v in defaults[choice].items():
    if isinstance(v, float):
        params[p] = st.sidebar.number_input(p, value=v, format="%.6f")
    else:
        params[p] = st.sidebar.number_input(p, value=v, step=1)
num_steps = st.sidebar.number_input("Number of Steps", 1, 500, 100, 1)

# Math formula display
st.subheader("📐 Mathematical Formulation")
formulae = {
    'Cosine Decay': r"LR(t)=α₀·(1+cos(π·t/T))/2",
    'Linear Cosine Decay': r"LR(t)=α₀·(α+(1−α)(1+cos(π·t/T))/2+β·t/T)",
    'Noisy Linear Cosine Decay': r"LR(t)=LR_{LCD}(t)+𝒩(0,σ²)",
    'Cyclical Learning Rate': r"c=⌊1+t/(2S)⌋,\;x=|t/S−2c+1|;\;LR=LR_{min}+(LR_{max}−LR_{min})·max(0,1−x)",
    'Exponential Cyclical Learning Rate': r"LR(t)=LR_{CLR}(t)·γ^{t}",
    'Custom Learning Scheduler': r"LR(t)=1/√{t+1}"
}
st.latex(formulae[choice])

# Plot LR schedule
steps = list(range(num_steps))
lrs = [schedulers[choice](step=s, **params) for s in steps]
fig_schedule = go.Figure([go.Scatter(x=steps, y=lrs, mode='lines')])
fig_schedule.update_layout(title=choice, xaxis_title='Step', yaxis_title='Learning Rate')
st.plotly_chart(fig_schedule, use_container_width=True)

# Code preview
st.subheader("📝 Scheduler Implementation")
code_snippet = f"""
{inspect.getsource(schedulers[choice])}

# Example usage
for step in range({num_steps}):
    lr = schedulers['{choice}'](step=step, **{params})
"""
st.code(code_snippet, language='python')

# ---------------- Training demo section (now for *any* scheduler) ----------------

st.subheader(f"🚀 Real‑Time Training Demo ({choice})")

import tensorflow as tf
if force_cpu:
    tf.config.set_visible_devices([], 'GPU')
else:
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

# Sidebar controls specific to demo
keras_epochs = st.sidebar.slider('Epochs', 1, 100, 5)
batch_size   = st.sidebar.selectbox('Batch Size', [16, 32, 64], 1)
train_samples = st.sidebar.slider('Train samples', 1000, 60000, 10000, 1000)
test_samples  = st.sidebar.slider('Test samples', 500, 10000, 2000, 500)

# Placeholders for live output
log_ph   = st.empty()
plot_ph  = st.empty()
prog_bar = st.progress(0)

# Keras callback for live logs
class StreamlitLogger(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.tr, self.va = [], []
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.tr.append(logs.get('accuracy', 0))
        self.va.append(logs.get('val_accuracy', 0))
        log_ph.markdown(f"**Epoch {epoch+1}/{self.total_epochs}** — "
                        f"loss: {logs.get('loss',0):.4f} — "
                        f"acc: {logs.get('accuracy',0):.4f} — "
                        f"val_acc: {logs.get('val_accuracy',0):.4f}")
        prog_bar.progress((epoch+1)/self.total_epochs)
        fig_rt = go.Figure([
            go.Scatter(x=list(range(1, epoch+2)), y=self.tr, mode='lines', name='Train'),
            go.Scatter(x=list(range(1, epoch+2)), y=self.va, mode='lines', name='Val')
        ])
        fig_rt.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy')
        plot_ph.plotly_chart(fig_rt, use_container_width=True, clear_figure=True)

# Build LR schedule function for Keras (epoch->lr)
def lr_fn(epoch):
    return schedulers[choice](step=epoch, **params)

if st.button('🎯 Start Training'):
    log_ph.info('Loading & subsampling MNIST…')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    rng = np.random.default_rng(0)
    idx_tr = rng.choice(x_train.shape[0], train_samples, replace=False)
    idx_te = rng.choice(x_test.shape[0],  test_samples,  replace=False)
    x_train, y_train = x_train[idx_tr]/255.0, y_train[idx_tr]
    x_test,  y_test  = x_test[idx_te]/255.0,  y_test[idx_te]
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    # Light CNN (~12 k params)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('initial_lr', 0.001)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=keras_epochs,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_fn),
                         StreamlitLogger(keras_epochs)],
              verbose=0)

    prog_bar.empty()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    st.metric('Final Test Accuracy', f"{acc*100:.2f}%")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        model.save(tmp.name)           # writes the .h5 file
        tmp.seek(0)                    # rewind to beginning
        st.download_button(
            label="💾 Download trained model (.h5)",
            data=tmp.read(),           # binary contents
            file_name="mnist_cnn.h5",  # what the user gets
            mime="application/octet-stream"
        )

    st.subheader('📉 Learning Rate per Epoch')
    lr_vals = [lr_fn(e) for e in range(keras_epochs)]
    fig_lr = go.Figure([go.Scatter(x=list(range(keras_epochs)), y=lr_vals, mode='lines')])
    fig_lr.update_layout(xaxis_title='Epoch', yaxis_title='LR')
    st.plotly_chart(fig_lr, use_container_width=True)

