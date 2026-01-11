import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df_history = pd.read_csv('./results/history.csv')

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


df_history.dropna

# Plot Accuracy
axes[0].plot(df_history.index + 1, df_history['accuracy'], label='Training Accuracy')
axes[0].plot(df_history.index + 1, df_history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot Loss
axes[1].plot(df_history.index + 1, df_history['loss'], label='Training Loss')
axes[1].plot(df_history.index + 1, df_history['val_loss'], label='Validation Loss')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

# Plot Learning Rate
axes[2].plot(df_history.index + 1, df_history['learning_rate'], label='Learning Rate', color='green')
axes[2].set_title('Learning Rate over Epochs')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('history_plots.png')