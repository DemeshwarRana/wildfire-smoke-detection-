# Wildfire Smoke Detection

An image classifier built with transfer learning (MobileNetV2, TensorFlow/Keras) that sorts a photo into one of three categories: **Smoke**, **fire**, or **non fire**.

## Problem Statement

Fire agencies and utility companies run networks of fixed cameras to watch for wildfires. The most valuable moment to catch a fire is while there is still only smoke and no visible flame, because that is the point where a fast response can still prevent major damage. Having a person watch every camera feed around the clock is slow and unreliable, so there is a real need for a system that can look at a single image and automatically tell smoke, fire, and ordinary scenery apart.

This is why the project is framed as a 3-class problem (Smoke / fire / non fire) instead of a simple fire-or-not binary classifier. A model that only reacts to visible flames is already too late. The evaluation in this project looks specifically at how often "Smoke" gets confused with "non fire" (a missed early warning, the costly mistake) versus with "fire" (a less costly mix-up that still results in a response being dispatched).

## Solution

The notebook (`wildfire_smoke_detection.ipynb`) walks through the full pipeline:

1. Downloading the dataset directly from Kaggle
2. Exploring the data (class balance, image sizes, sample images)
3. Auditing the dataset for quality problems
4. Building a data loading pipeline
5. Training a MobileNetV2-based classifier using transfer learning
6. Evaluating the model on a held-out test set
7. Testing the model on real photographs from outside the dataset
8. Using Grad-CAM to visualize what the model is actually focusing on
9. A written discussion of limitations and future work

## Dataset

**[Forest Fire, Smoke, and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)** (Kaggle), roughly 6 GB, 42,900 images split evenly across three classes: fire, Smoke, and non fire.

## A key finding: the TIFF problem

While auditing the dataset, a check using TensorFlow's own image decoder revealed that a large number of files could not actually be read by the standard `image_dataset_from_directory` loading method: about 1,056 test images and 692 training images failed to decode, and **every single one of them belonged to the Smoke class**.

The cause: these files were saved in TIFF format, which TensorFlow's built-in decoder does not support, even though the files are perfectly valid images. An earlier, simpler check using PIL had reported the dataset as having no corrupted files at all, which was misleading, since PIL can open TIFF files without any trouble while TensorFlow cannot.

This mattered because Smoke is the exact class this project cares about most. Left unnoticed, the model would have trained and been evaluated on a smaller, silently incomplete version of the Smoke class. The fix was to stop relying on TensorFlow's built-in decoder entirely and build a custom loading pipeline using PIL (which reads any common image format, including TIFF), so every image in the dataset could actually be used.

## Results

On the full, corrected test set (10,500 images):

| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| Smoke    | 0.98      | 0.97   | 0.98     |
| fire     | 0.97      | 0.98   | 0.98     |
| non fire | 0.97      | 0.98   | 0.98     |

**Overall test accuracy: 97.8%**

Of 3,500 real Smoke test images: 3,407 were correctly identified, 49 were missed as "non fire" (the costly error, about 1.4%), and 44 were mistaken for "fire" (the less costly error, about 1.3%).

## External validation

To check whether the model generalizes beyond this dataset's own style of photography, it was tested on two real photographs found outside the dataset entirely:

- A **thin, distant haze under a clear blue sky** (close to the dataset's own typical "Smoke" style) was correctly classified as **Smoke**, with 87% confidence.
- A **thick, dramatic plume with a warm orange glow** was misclassified as **fire** (69% fire vs. 31% Smoke), though "non fire" was correctly ruled out almost entirely (0.2%).

Using Grad-CAM to visualize what the model focused on for the second image showed it was reacting to the warm orange glow at the base of the plume rather than the shape of the smoke itself, suggesting the training data underrepresents thick, close-range smoke with warm lighting.

## Limitations

- The dataset is made up of individually curated photographs, not continuous footage from a real fixed monitoring camera, so a real deployed system would likely see images that differ in distance, angle, and lighting.
- The model was tested on only two real-world external photographs, enough to reveal a pattern but not a thorough evaluation.
- This project classifies single still images. A real early-warning system would need to process a continuous video stream and decide when smoke has persisted long enough to be worth alerting someone about.

## Future Work

- Collect or source more training examples of thick, close-range smoke with warm lighting, so the model stops associating that appearance with fire specifically.
- Fine-tune the deeper layers of MobileNetV2 instead of only training the classification head.
- Test the model on footage from a real wildfire camera network, such as the publicly available HPWREN archive.
- Extend the project from single-image classification to object detection, so the system can localize where in the frame the smoke actually is, and eventually to a video-based pipeline that tracks a camera feed over time.

## What's in this repo

- `wildfire_smoke_detection.ipynb` — the full pipeline described above, with explanatory markdown throughout.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT license.

The dataset itself is not committed here; the notebook downloads it directly.

## Setup and how to run

```bash
git clone https://github.com/DemeshwarRana/wildfire-smoke-detection-.git
cd wildfire-smoke-detection-
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

This notebook was built and run on **Google Colab** with a T4 GPU, since the dataset is about 6 GB and a GPU significantly speeds up training. To reproduce it:

1. Open the notebook in Google Colab (**File → Upload notebook**, or open directly from GitHub via **File → Open notebook → GitHub** and paste this repo's URL).
2. Get a Kaggle API token: kaggle.com → profile picture → **Settings** → **API** → **Create New Token**. This gives you a username and a key.
3. In Colab, click the key icon in the left sidebar (**Secrets**) and add two secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY`, using the values from step 2. Toggle notebook access on for both.
4. **Runtime → Change runtime type → T4 GPU**.
5. **Runtime → Run all.** The notebook downloads the dataset directly from Kaggle using `kagglehub`, so no manual download is needed.

## References

- Amerzish Minha. *Forest Fire, Smoke, and Non-Fire Image Dataset*. Kaggle.
- HPWREN (High Performance Wireless Research and Education Network). Wildfire camera archive, UC San Diego.
- Sandler, M. et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*.
- Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*.
