# lr-scheduler-playground
Interactive Learning Rate Scheduler Playground built with Streamlit and TensorFlow. Visualize, experiment, and train lightweight models with custom LR schedules in real time.

# ⚙️ LR Scheduler Playground

A minimal Streamlit app to visualize and test popular **learning rate schedules** with live training on MNIST (using TensorFlow).


https://github.com/user-attachments/assets/0c684705-c5d0-4ec0-8d42-f483a59832a7


## 🚀 Setup

```bash
git clone https://github.com/yllberisha/lr-scheduler-playground.git
cd lr-scheduler-playground
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run lr_scheduler_app.py
