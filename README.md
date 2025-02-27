# SEMANTIC-SEGMENTATION-IN-AUTONOMOUS-VEHICLES

## 📌 Project Overview
This project focuses on **real-time semantic segmentation** for autonomous vehicles using deep learning. The goal is to classify each pixel in an image to identify road elements such as **vehicles, pedestrians, lanes, and obstacles**.

## 📂 Dataset
We use the **CamVid Dataset**, which contains labeled video sequences of driving scenes.
- **Train Images**: `CamVid/train`
- **Train Masks**: `CamVid/trainannot`
- **Validation Images**: `CamVid/val`
- **Validation Masks**: `CamVid/valannot`
- **Test Images**: `CamVid/test`
- **Test Masks**: `CamVid/testannot`

## 📦 Installation & Setup
Clone the repository and install dependencies:
```bash
# Clone repository
git clone https://github.com/MokshagnaAnurag/SEMANTIC-SEGMENTATION-IN-AUTONOMOUS-VEHICLES.git
cd semantic-segmentation-av

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Model Training
Train the model using **DeepLabV3+** with a **ResNet-50** backbone:
```python
python train.py --epochs 10 --batch_size 8 --lr 0.001
```

## 🏆 Evaluation Metrics
We evaluate the model using:
- **IoU (Intersection over Union)**
- **Dice Coefficient**
- **Pixel Accuracy**

```python
python evaluate.py --model saved_model.pth
```

## 🎥 Real-Time Inference on Video
Run inference on a live video feed or pre-recorded video:
```python
python inference.py --video test_video.mp4 --model saved_model.pth
```

## 🖥️ Deployment
You can deploy the model using **ONNX Runtime** or on **Jetson Nano** for edge inference:
```bash
python export_onnx.py --model saved_model.pth
```

## 📜 Results
| Metric  | Score  |
|---------|--------|
| IoU     | 0.87   |
| Dice    | 0.89   |
| Accuracy| 92.3%  |

## 📌 Future Improvements
- Improve segmentation under **low-light and foggy conditions**.
- Optimize the model for **real-time inference on embedded systems**.
- Experiment with **Transformer-based architectures (Segment Anything Model - SAM)**.

## 🤝 Contributing
If you’d like to contribute:
1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit changes and push
4. Submit a pull request

## 📜 License
This project is licensed under the MIT License.


