# 🧠 Hopfield Network for Pattern Recognition

A complete implementation of a Hopfield Neural Network for pattern recognition with image upload capabilities and a beautiful web interface.

## 📌 Overview

This project implements a Hopfield Neural Network that can:
- Train on predefined binary patterns (letters P and Q)
- Accept noisy or distorted images as input
- Recall the closest stored pattern using associative memory
- Provide both web interface and interactive demo

## 🏗️ Project Structure

```
Hopfield_Project/
│── hopfield.py           # Core Hopfield network implementation
│── app.py               # Flask web application
│── requirements.txt     # Python dependencies
│── README.md           # This file
│── templates/
│   ├── index.html      # Main upload page
│   ├── results.html    # Results display page
│   ├── demo.html       # Interactive demo page
│   ├── 404.html        # Error page
│   └── 500.html        # Server error page
│── static/             # Generated visualizations
│   └── result.png      # Latest pattern comparison
└── uploads/            # Temporary uploaded files
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy matplotlib pillow flask werkzeug
```

## 🚀 Running the Application

### Start the Web Server
```bash
python app.py
```

The application will be available at: `http://localhost:5000`

### Test the Core Network
```bash
python hopfield.py
```

This will run a test of the Hopfield network with noisy patterns and generate visualization files.

## 🧠 How It Works

### Hopfield Network Theory

The Hopfield Network is a recurrent neural network that stores patterns as stable states in its weight matrix.

#### 1. Training (Hebbian Learning)
```python
for pattern in patterns:
    W += np.outer(pattern, pattern)
np.fill_diagonal(W, 0)  # No self-connections
```

Mathematical representation:
```
W_ij = Σ_μ x_i^(μ) * x_j^(μ), W_ii = 0
```

#### 2. Recall (Asynchronous Updates)
```python
for i in range(n):
    net_input = np.dot(W[i], state)
    state[i] = 1 if net_input >= 0 else -1
```

Mathematical representation:
```
s_i(t+1) = sign(Σ_j W_ij * s_j(t))
```

#### 3. Image Preprocessing
```python
img = Image.open(path).convert("L")      # Grayscale
img = img.resize((5,5))                  # 5x5 grid
binary = np.where(arr < 128, 1, -1)     # Threshold
vector = binary.flatten()               # Flatten to 1D
```

### Training Patterns

The network is pre-trained on two 5×5 binary patterns:

**Pattern P:**
```
1  1  1  1 -1
1 -1 -1 -1  1
1  1  1  1 -1
1 -1 -1 -1 -1
1 -1 -1 -1 -1
```

**Pattern Q:**
```
-1  1  1  1 -1
 1 -1 -1 -1  1
 1 -1 -1 -1  1
 1 -1 -1  1  1
-1  1  1  1  1
```

## 🌐 Web Interface Features

### 1. Main Upload Page (`/`)
- Beautiful, responsive design
- File upload with drag-and-drop support
- Visual display of training patterns
- File type validation and size limits

### 2. Results Page (`/upload`)
- Side-by-side comparison of input, recalled, and stored patterns
- Detailed similarity scores
- Convergence information and iteration count
- Technical details about the network

### 3. Interactive Demo (`/demo`)
- Click-to-edit 5×5 pattern grid
- Real-time pattern recognition
- Preset patterns and noise generation
- Instant feedback with similarity scores

### 4. API Endpoints
- `GET /api/patterns` - Get training patterns
- `POST /api/recall` - Recall pattern from JSON input

## 📊 Usage Examples

### Web Interface Usage

1. **Upload an Image:**
   - Go to `http://localhost:5000`
   - Click "Choose File" and select an image
   - Click "Recognize Pattern"
   - View results with detailed analysis

2. **Interactive Demo:**
   - Go to `http://localhost:5000/demo`
   - Click cells to create patterns
   - Use preset patterns or add noise
   - Click "Recall Pattern" to see results

3. **Test with Sample Images:**
   - Use the provided sample images in `noisy_samples/` folder
   - Try different noise levels: clean, blur, salt & pepper
   - Compare how the network handles different types of degradation

### 🧪 Sample Test Images

The `noisy_samples/` folder contains test images for evaluating network performance:

| Image | Description | Expected Result |
|-------|-------------|-----------------|
| `P_clean.png` | Clean P pattern | ~100% accuracy |
| `P_blur.png` | Blurred P pattern | Good accuracy |
| `P_saltpepper.png` | Noisy P pattern | Moderate accuracy |
| `Q_clean.png` | Clean Q pattern | ~100% accuracy |
| `Q_blur.png` | Blurred Q pattern | Good accuracy |
| `Q_saltpepper.png` | Noisy Q pattern | Moderate accuracy |

### Programmatic Usage

```python
from hopfield import HopfieldNetwork, create_training_patterns

# Initialize network
patterns = create_training_patterns()
network = HopfieldNetwork(size=25)
network.train(list(patterns.values()))

# Create noisy input
noisy_input = add_noise_to_pattern(patterns['P'], noise_level=0.2)

# Recall pattern
recalled, converged, iterations = network.recall(noisy_input)
print(f"Converged: {converged}, Iterations: {iterations}")

# Calculate similarity
similarity = network.pattern_similarity(recalled, patterns['P'])
print(f"Similarity to P: {similarity:.2f}")
```

## 🎯 Key Features

### Core Network
- ✅ Hebbian learning rule implementation
- ✅ Asynchronous neuron updates
- ✅ Energy function calculation
- ✅ Pattern similarity metrics
- ✅ Convergence detection

### Image Processing
- ✅ Automatic image preprocessing
- ✅ Resize to 5×5 grid
- ✅ Grayscale conversion
- ✅ Binary thresholding
- ✅ Support for multiple image formats

### Web Interface
- ✅ Modern, responsive design
- ✅ File upload with validation
- ✅ Real-time pattern editing
- ✅ Beautiful visualizations
- ✅ Error handling and user feedback

### Visualization
- ✅ Pattern comparison plots
- ✅ Similarity bar charts
- ✅ Interactive grid displays
- ✅ Color-coded results

## 🔧 Configuration

### Network Parameters
- **Grid Size:** 5×5 (25 neurons)
- **Patterns:** P and Q letters
- **Update Rule:** Asynchronous
- **Activation:** Sign function
- **Learning:** Hebbian rule

### Image Processing
- **Target Size:** 5×5 pixels
- **Threshold:** 128 (grayscale)
- **Supported Formats:** PNG, JPG, JPEG, GIF, BMP, TIFF
- **Max File Size:** 16MB

### Web Server
- **Host:** 0.0.0.0 (all interfaces)
- **Port:** 5000
- **Debug Mode:** Enabled in development

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   pip install --upgrade numpy matplotlib pillow flask
   ```

2. **Port Already in Use:**
   - Change port in `app.py`: `app.run(port=5001)`
   - Or kill existing process: `lsof -ti:5000 | xargs kill`

3. **Image Processing Errors:**
   - Ensure image file is not corrupted
   - Try different image formats
   - Check file size (max 16MB)

4. **Network Not Converging:**
   - This is normal for very noisy inputs
   - Try reducing noise level
   - Check pattern similarity scores

### Performance Tips

- Use smaller images for faster processing
- Clear browser cache if interface issues occur
- Restart Flask server if memory usage is high

## 📚 Technical Details

### Energy Function
The network minimizes the energy function:
```
E = -½ Σᵢⱼ wᵢⱼ sᵢ sⱼ
```

### Capacity
Theoretical capacity: ~0.15N patterns (N = number of neurons)
- For 25 neurons: ~3-4 patterns maximum
- Current implementation: 2 patterns (well within capacity)

### Convergence
- Guaranteed for stored patterns (noise-free)
- May converge to spurious states with noise
- Maximum iterations: 100 (configurable)

## 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Educational Use

This implementation is perfect for:
- Neural network courses
- Pattern recognition studies
- Machine learning demonstrations
- Interactive learning experiences

## 🔗 References

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities.
- Hertz, J., Krogh, A., & Palmer, R. G. (1991). Introduction to the theory of neural computation.

---

**Happy Pattern Recognition! 🧠✨**
