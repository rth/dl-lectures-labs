# Keras 3 + PyTorch Backend Migration Plan

## Overview

Migrate Jupyter notebooks and Python files from TensorFlow 2.x/tensorflow.keras to Keras 3.x with PyTorch backend.
- Update to Python 3.12 (latest stable)
- Use `uv` for environment management
- Skip `_rendered` notebooks (can be regenerated after migration)

## Migration Summary

| Issue | Count | Severity |
|-------|-------|----------|
| `tensorflow.keras` → `keras` imports | ~20 notebooks + ~15 .py files | Medium |
| `fit_generator()` deprecated | 1 notebook | Critical |
| `ImageDataGenerator` incompatible | 1 notebook | Critical |
| `Convolution2D` → `Conv2D` | 2 notebooks + 2 .py files | Medium |
| `lr` → `learning_rate` | 6 notebooks + ~5 .py files | Low |
| Custom TF ops in layers | 1 notebook + 1 .py file | High |
| Python 3.9 → 3.12 | All files | Medium |

---

## Phase 0: Environment Setup with uv

### Step 1: Create virtual environment with uv

```bash
cd /home/rth/src/dl-lectures-labs

# Create venv with Python 3.12
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate
```

### Step 2: Update requirements.txt

**File:** `/home/rth/src/dl-lectures-labs/requirements.txt`

```
# Core dependencies
numpy
scipy
scikit-learn
jupyter
matplotlib
pandas
pillow
scikit-image
lxml
opencv-python
setuptools>=41.0.0

# Keras 3 with PyTorch backend
keras>=3.0.0
torch>=2.0.0
torchvision>=0.15.0

# h5py - no longer pinned (Keras 3 prefers .keras format)
h5py>=3.0.0
```

### Step 3: Update runtime.txt

**File:** `/home/rth/src/dl-lectures-labs/runtime.txt`

```
python-3.12
```

### Step 4: Update environment.yml (if keeping conda support)

**File:** `/home/rth/src/dl-lectures-labs/environment.yml`

```yaml
name: dlclass2
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.12
  - pip:
    - -r requirements.txt
```

### Step 5: Install dependencies and verify

```bash
# Install all dependencies
uv pip install -r requirements.txt

# Verify Keras 3 with PyTorch backend
python -c "import os; os.environ['KERAS_BACKEND']='torch'; import keras; print(f'Keras {keras.__version__} with PyTorch backend')"

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Register Jupyter kernel
python -m ipykernel install --user --name=dlclass2 --display-name="DL Class (Keras 3 + PyTorch)"
```

---

## Phase 1: Lab 01 - Keras Intro (Foundation)

**Priority: HIGHEST** - Establishes patterns for all other notebooks

### Notebooks to update:
1. `labs/01_keras/Intro Keras.ipynb`
2. `labs/01_keras/data_download.ipynb`

### Python files to update:
1. `labs/01_keras/solutions/keras_adam.py`
2. `labs/01_keras/solutions/keras_sgd_and_momentum.py`
3. `labs/01_keras/solutions/inception_resnet_v2.py`

### Changes Pattern (apply to ALL files):

```python
# Add as FIRST lines in every .py file and first code cell in notebooks:
import os
os.environ["KERAS_BACKEND"] = "torch"

# Change all imports from:
from tensorflow.keras.X import Y
# To:
from keras.X import Y
```

### Specific Import Changes:
| Old | New |
|-----|-----|
| `from tensorflow.keras.utils import to_categorical` | `from keras.utils import to_categorical` |
| `from tensorflow.keras.models import Sequential` | `from keras.models import Sequential` |
| `from tensorflow.keras.layers import Dense` | `from keras.layers import Dense` |
| `from tensorflow.keras import optimizers` | `from keras import optimizers` |
| `from tensorflow.keras.callbacks import TensorBoard` | `from keras.callbacks import TensorBoard` |
| `from tensorflow.keras import initializers` | `from keras import initializers` |

### Verification:
```bash
cd /home/rth/src/dl-lectures-labs
jupyter nbconvert --to notebook --execute labs/01_keras/Intro\ Keras.ipynb --output /tmp/test_output.ipynb
```
- [ ] All cells execute without errors
- [ ] Model compiles and trains for 15 epochs
- [ ] Accuracy metrics display (~97% on MNIST)

---

## Phase 2: Lab 03 - Neural Recommender Systems

**Priority: HIGH** - Core embeddings content

### Notebooks to update:
1. `labs/03_neural_recsys/Short_Intro_to_Embeddings_with_Keras.ipynb`
2. `labs/03_neural_recsys/Explicit_Feedback_Neural_Recommender_System.ipynb`
3. `labs/03_neural_recsys/Implicit_Feedback_Recsys_with_the_triplet_loss.ipynb`
4. `labs/03_neural_recsys/data_download.ipynb`

### Python files to update:
1. `labs/03_neural_recsys/solutions/deep_implicit_feedback_recsys.py`
2. `labs/03_neural_recsys/solutions/embeddings_sequential_model.py`
3. `labs/03_neural_recsys/movielens_paramsearch.py`

### Changes:
- Standard import changes (tensorflow.keras → keras)
- Add backend setup cell/lines
- Fix any `merge` layer usage (deprecated) → use `Concatenate`, `Add`, etc.

### Verification:
- [ ] Embedding layers work correctly
- [ ] Custom Model subclasses function
- [ ] Training converges on MovieLens data

---

## Phase 3: Lab 04 - Conv Nets (CRITICAL)

**Priority: CRITICAL** - Contains deprecated `fit_generator` and `ImageDataGenerator`

### Notebooks to update:
1. `labs/04_conv_nets/01_Convolutions.ipynb` - Standard changes
2. `labs/04_conv_nets/02_Pretrained_ConvNets_with_Keras.ipynb` - Standard changes
3. `labs/04_conv_nets/03_Fine_Tuning_Deep_CNNs_with_GPU.ipynb` - **MAJOR REWRITE**
4. `labs/04_conv_nets/04_ConvNet_Design.ipynb` - `lr` → `learning_rate`

### Python files to update:
1. `labs/04_conv_nets/solutions/mini_resnet.py`
2. `labs/04_conv_nets/solutions/mini_inception.py`
3. `labs/04_conv_nets/solutions/predict_image.py`
4. `labs/04_conv_nets/process_images.py`

### Critical File: `03_Fine_Tuning_Deep_CNNs_with_GPU.ipynb`

#### A. ImageDataGenerator Replacement

**Current (incompatible with PyTorch backend):**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
augmenting_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    channel_shift_range=9,
    fill_mode='nearest'
)
train_flow = augmenting_datagen.flow_from_directory(...)
```

**Replace with PyTorch DataLoader:**
```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_folder, transform=train_transforms)
val_dataset = datasets.ImageFolder(validation_folder, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

#### B. fit_generator Replacement

**Current (deprecated):**
```python
history = model.fit_generator(train_flow, 5000,
                              epochs=30,
                              validation_data=val_flow,
                              validation_steps=val_flow.n)
```

**Replace with:**
```python
history = model.fit(train_loader,
                    epochs=30,
                    validation_data=val_loader)
```

#### C. Learning rate parameter

**Current:**
```python
opt = optimizers.SGD(lr=1e-4, momentum=0.9)
top_model.compile(optimizer=Adam(lr=1e-4), ...)
```

**Replace with:**
```python
opt = optimizers.SGD(learning_rate=1e-4, momentum=0.9)
top_model.compile(optimizer=Adam(learning_rate=1e-4), ...)
```

### Verification:
- [ ] PyTorch DataLoader loads images correctly
- [ ] Data augmentation works visually (display augmented images)
- [ ] ResNet50 pretrained model loads from keras.applications
- [ ] Fine-tuning trains without errors
- [ ] Validation accuracy ~98-99% on dogs-vs-cats

---

## Phase 4: Lab 05 - Conv Nets 2 (CRITICAL)

**Priority: CRITICAL** - Contains `Convolution2D` and custom TF layers

### Notebooks to update:
1. `labs/05_conv_nets_2/ConvNets_for_Classification_and_Localization.ipynb`
2. `labs/05_conv_nets_2/Fully_Convolutional_Neural_Networks.ipynb`
3. `labs/05_conv_nets_2/data_download.ipynb`

### Python files to update:
1. `labs/05_conv_nets_2/solutions/fully_conv.py`
2. `labs/05_conv_nets_2/solutions/classif_and_loc.py`
3. `labs/05_conv_nets_2/solutions/load_pretrained.py`
4. `labs/05_conv_nets_2/compute_representations.py`
5. `labs/05_conv_nets_2/get_dense_weights.py`

### Key Changes:

#### A. Convolution2D → Conv2D
```python
# From:
from tensorflow.keras.layers import Convolution2D
x = Convolution2D(4, 1, 1, activation='relu')(input)

# To:
from keras.layers import Conv2D
x = Conv2D(4, (1, 1), activation='relu')(input)
```

#### B. Custom Layer with TF ops → keras.ops

**Current (Fully_Convolutional_Neural_Networks.ipynb):**
```python
import tensorflow as tf

class SoftmaxMap(layers.Layer):
    def call(self, x, mask=None):
        e = tf.exp(x - tf.math.reduce_max(x, axis=self.axis, keepdims=True))
        s = tf.math.reduce_sum(e, axis=self.axis, keepdims=True)
        return e / s
```

**Replace with backend-agnostic keras.ops:**
```python
from keras import ops

class SoftmaxMap(layers.Layer):
    def call(self, x, mask=None):
        e = ops.exp(x - ops.max(x, axis=self.axis, keepdims=True))
        s = ops.sum(e, axis=self.axis, keepdims=True)
        return e / s
```

#### C. scipy.misc removal (if present)
```python
# From:
from scipy.misc import imread, imresize

# To:
from PIL import Image
import numpy as np

def imread(path):
    return np.array(Image.open(path))

def imresize(img, size):
    return np.array(Image.fromarray(img).resize(size))
```

### Verification:
- [ ] Conv2D layers work correctly
- [ ] Custom SoftmaxMap layer functions with PyTorch backend
- [ ] Pretrained model inference works
- [ ] Classification and localization outputs correct

---

## Phase 5: Lab 07 - Deep NLP

**Priority: HIGH** - Contains `lr` parameter issues

### Notebooks to update:
1. `labs/07_deep_nlp_2/Character_Level_Language_Model.ipynb`
2. `labs/07_deep_nlp_2/NLP_word_vectors_classification.ipynb`
3. `labs/07_deep_nlp_2/data_download.ipynb`

### Python files to update:
1. `labs/07_deep_nlp_2/solutions/lstm.py`
2. `labs/07_deep_nlp_2/solutions/conv1d.py`

### Changes:
- Standard import changes (tensorflow.keras → keras)
- `lr=0.01` → `learning_rate=0.01` in RMSprop and Adam
- `keras.preprocessing.text.Tokenizer` → check if still available or use keras-nlp

### Text Preprocessing Note:
```python
# Keras 3 may move preprocessing - check availability:
try:
    from keras.preprocessing.text import Tokenizer
    from keras.utils import pad_sequences
except ImportError:
    # Alternative: use keras_nlp or custom implementation
    from keras_nlp.tokenizers import Tokenizer
```

### Verification:
- [ ] LSTM/GRU layers work with PyTorch backend
- [ ] Text tokenization works
- [ ] Character-level model generates text
- [ ] Word vectors classification trains

---

## Phase 6: Lab 08 - Seq2Seq

**Priority: HIGH** - Model saving/loading

### Notebooks to update:
1. `labs/08_seq2seq/Translation_of_Numeric_Phrases_with_Seq2Seq.ipynb`
2. `labs/08_seq2seq/data_download.ipynb`

### Python files to update:
1. `labs/08_seq2seq/solutions/simple_seq2seq.py`

### Changes:
- Standard import changes
- Model save format: `.h5` → `.keras` (recommended for Keras 3)

```python
# From:
best_model_fname = "simple_seq2seq_checkpoint.h5"
model.save(best_model_fname)
model = load_model(best_model_fname)

# To:
best_model_fname = "simple_seq2seq_checkpoint.keras"
model.save(best_model_fname)
model = keras.saving.load_model(best_model_fname)
```

### Verification:
- [ ] Seq2Seq model trains with GRU layers
- [ ] Model saves in .keras format
- [ ] Model loads correctly
- [ ] Inference produces valid translations

---

## Phase 7: Lab 02 - Backpropagation

**Priority: LOW** - Mixed content, mostly educational

### Notebooks:
1. `labs/02_backprop/Backpropagation_numpy.ipynb` - **No changes needed** (pure NumPy)
2. `labs/02_backprop/Backpropagation_tensorflow.ipynb` - **Keep as-is** (TF educational content)
3. `labs/02_backprop/Backpropagation_pytorch.ipynb` - **No changes needed** (already PyTorch)

### Python files to update:
1. `labs/02_backprop/solutions/keras_model.py` - Update imports, `lr` → `learning_rate`

### Verification:
- [ ] keras_model.py solution works with Keras 3

---

## Phase 8: Lab 06 - Classical NLP

**Priority: NONE** - No Keras content

All notebooks use sklearn only:
- `document_classification_20newsgroups.ipynb`
- `trec-news-dataset-search.ipynb`
- `regex_tutorial_exercise_questions.ipynb`
- `nlp-topic-modelling-of-movie-plots-with-spacy.ipynb`

**No migration needed.**

---

## Complete File List

### Notebooks to migrate (18 total):

| Lab | Notebook | Priority |
|-----|----------|----------|
| 01 | `Intro Keras.ipynb` | HIGHEST |
| 01 | `data_download.ipynb` | HIGH |
| 03 | `Short_Intro_to_Embeddings_with_Keras.ipynb` | HIGH |
| 03 | `Explicit_Feedback_Neural_Recommender_System.ipynb` | HIGH |
| 03 | `Implicit_Feedback_Recsys_with_the_triplet_loss.ipynb` | HIGH |
| 03 | `data_download.ipynb` | MEDIUM |
| 04 | `01_Convolutions.ipynb` | HIGH |
| 04 | `02_Pretrained_ConvNets_with_Keras.ipynb` | HIGH |
| 04 | `03_Fine_Tuning_Deep_CNNs_with_GPU.ipynb` | CRITICAL |
| 04 | `04_ConvNet_Design.ipynb` | HIGH |
| 05 | `ConvNets_for_Classification_and_Localization.ipynb` | CRITICAL |
| 05 | `Fully_Convolutional_Neural_Networks.ipynb` | CRITICAL |
| 05 | `data_download.ipynb` | MEDIUM |
| 07 | `Character_Level_Language_Model.ipynb` | HIGH |
| 07 | `NLP_word_vectors_classification.ipynb` | HIGH |
| 07 | `data_download.ipynb` | MEDIUM |
| 08 | `Translation_of_Numeric_Phrases_with_Seq2Seq.ipynb` | HIGH |
| 08 | `data_download.ipynb` | MEDIUM |

### Python files to migrate (18 total):

| Lab | File | Priority |
|-----|------|----------|
| 01 | `solutions/keras_adam.py` | HIGH |
| 01 | `solutions/keras_sgd_and_momentum.py` | HIGH |
| 01 | `solutions/inception_resnet_v2.py` | MEDIUM |
| 02 | `solutions/keras_model.py` | LOW |
| 03 | `solutions/deep_implicit_feedback_recsys.py` | HIGH |
| 03 | `solutions/embeddings_sequential_model.py` | HIGH |
| 03 | `movielens_paramsearch.py` | MEDIUM |
| 04 | `solutions/mini_resnet.py` | HIGH |
| 04 | `solutions/mini_inception.py` | HIGH |
| 04 | `solutions/predict_image.py` | MEDIUM |
| 04 | `process_images.py` | MEDIUM |
| 05 | `solutions/fully_conv.py` | CRITICAL |
| 05 | `solutions/classif_and_loc.py` | CRITICAL |
| 05 | `solutions/load_pretrained.py` | HIGH |
| 05 | `compute_representations.py` | MEDIUM |
| 05 | `get_dense_weights.py` | MEDIUM |
| 07 | `solutions/lstm.py` | HIGH |
| 07 | `solutions/conv1d.py` | HIGH |
| 08 | `solutions/simple_seq2seq.py` | HIGH |

---

## Execution Order Summary

1. **Phase 0**: Environment setup with uv (Python 3.12, requirements.txt)
2. **Phase 1**: Lab 01 - Foundation patterns (2 notebooks + 3 .py files)
3. **Phase 2**: Lab 03 - Embeddings (4 notebooks + 3 .py files)
4. **Phase 3**: Lab 04 - Conv nets + ImageDataGenerator rewrite (4 notebooks + 4 .py files)
5. **Phase 4**: Lab 05 - Conv nets 2 + keras.ops (3 notebooks + 5 .py files)
6. **Phase 5**: Lab 07 - NLP (3 notebooks + 2 .py files)
7. **Phase 6**: Lab 08 - Seq2Seq (2 notebooks + 1 .py file)
8. **Phase 7**: Lab 02 - Backprop solution file (1 .py file)
9. **Phase 8**: Lab 06 - Verify no changes needed

---

## Verification Protocol

### Per-notebook verification:
```bash
# Activate environment
source .venv/bin/activate

# Execute notebook and check for errors
jupyter nbconvert --to notebook --execute <notebook.ipynb> --output /tmp/test.ipynb

# Or run interactively
jupyter lab
```

### Checklist per file:
- [ ] Backend setup present (os.environ["KERAS_BACKEND"] = "torch")
- [ ] All tensorflow.keras imports changed to keras
- [ ] All `lr=` changed to `learning_rate=`
- [ ] All `Convolution2D` changed to `Conv2D`
- [ ] All `fit_generator` changed to `fit`
- [ ] All tf.* ops in custom layers changed to keras.ops.*
- [ ] File executes without import errors
- [ ] Models train successfully

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ImageDataGenerator incompatible | Full rewrite with PyTorch DataLoader + torchvision transforms |
| Pre-trained .h5 models incompatible | Re-save in .keras format or provide conversion script |
| Custom TF layers | Convert to keras.ops for backend-agnostic code |
| keras.preprocessing.text changes | Check keras-nlp or provide custom implementation |
| Python 3.12 compatibility | Test all notebooks with new Python version |
| GPU-dependent notebooks | Document GPU requirements clearly |
