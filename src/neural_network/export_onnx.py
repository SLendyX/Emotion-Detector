import torch
import torch.onnx
import onnxruntime as ort
import time
import numpy as np
from my_training import SimpleEmotionCNN


# --- CONFIG ---
MODEL_PATH = "models/emotion_model_epoch_50.pt" # Calea către modelul antrenat
ONNX_PATH = "models/final_model.onnx"
DEVICE = torch.device("cpu") # Măsurăm latența pe CPU (scenariu realist)

def main():
    # 1. Încărcăm modelul PyTorch
    print("🔄 Încărcare model PyTorch...")
    model = SimpleEmotionCNN(num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Creăm un input dummy (simulăm o imagine 100x100)
    dummy_input = torch.randn(1, 3, 100, 100)

    # 3. Exportăm în ONNX
    print(f"📦 Exportare în {ONNX_PATH}...")
    torch.onnx.export(model, 
                      dummy_input, 
                      ONNX_PATH, 
                      input_names=['input'], 
                      output_names=['output'],)
    print("✅ Export reușit!")

    # 4. Benchmark Latență (Viteză)
    print("\n⏱️  Rulez Benchmark Latență (ONNX Runtime)...")
    ort_session = ort.InferenceSession(ONNX_PATH)
    
    # Pregătim datele pentru ONNX (numpy array)
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

    # Warm-up (câteva rulări de încălzire)
    for _ in range(10):
        ort_session.run(None, onnx_input)

    # Măsurăm 100 de rulări
    start_time = time.time()
    num_runs = 100
    for _ in range(num_runs):
        ort_session.run(None, onnx_input)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000 # convertim în ms
    print(f"🚀 Timp mediu per inferență: {avg_time:.2f} ms")

    if avg_time < 50:
        print("✅ Obiectiv atins: < 50ms")
    else:
        print("⚠️ Obiectiv ratat: > 50ms")

if __name__ == "__main__":
    main()