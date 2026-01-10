import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import h5py
from PIL import Image
import numpy as np

model = ResNet50(include_top=True, weights='imagenet')
input = model.layers[0].input
output = model.layers[-2].output
base_model = Model(inputs=input, outputs=output)
del model


paths = ["images_resize/" + path for path in sorted(os.listdir("images_resize/"))]
batch_size = 32
out_tensors = np.zeros((len(paths), 2048), dtype="float32")
print(out_tensors.shape)
for idx in range(len(paths) // batch_size + 1):
    batch_bgn = idx * batch_size
    batch_end = min((idx+1) * batch_size, len(paths))
    imgs = []
    for path in paths[batch_bgn:batch_end]:
        img = np.array(Image.open(path))
        img = np.array(Image.fromarray(img).resize((224, 224))).astype("float32")
        img = preprocess_input(img[np.newaxis])
        imgs.append(img)
    batch_tensor = np.vstack(imgs)
    print("tensor", idx, "with shape",batch_tensor.shape)
    out_tensor = base_model.predict(batch_tensor, batch_size=32)
    print("output shape:", out_tensor.shape)
    out_tensors[batch_bgn:batch_end, :] = out_tensor
print("shape of representation", out_tensors.shape)

# Serialize representations
h5f = h5py.File('img_emb.h5', 'w')
h5f.create_dataset('img_emb', data=out_tensors)
h5f.close()
