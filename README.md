# **Mini GACS Prototype – Mood & Style Embedding Pipeline**

**GenTA Competency Assessment – AI R&D Engineer**



## **Overview**

This project implements a small end-to-end affective computing prototype inspired by GenTA’s GACS framework. The system extracts frames from art/marketing videos, computes multimodal embeddings using CLIP, and analyzes similarity to estimate visual mood and style relationships between creatives.

The objective is to demonstrate a **verification-first AI R&D workflow** for representing creative “vibe” in embedding space using modern pretrained multimodal models.



## **Pipeline Summary**

The notebook follows a structured pipeline:

1. **Video Ingestion**
   Short royalty-free art or marketing-style videos are treated as creative assets.

2. **Frame Extraction**
   Frames are sampled at fixed time intervals using OpenCV.

3. **Metadata Tracking**
   A CSV file stores:

   * `video_id`
   * `frame_id`
   * `file_path`

   This ensures reproducibility across runs.

4. **Embedding Generation (CLIP)**
   Frames are encoded using **CLIP (ViT-B/32)** to produce **512-dimensional embeddings** capturing aesthetic and stylistic features such as composition, color, and texture.

5. **Similarity Analysis**

   * Pairwise cosine similarity matrix
   * Similarity heatmap visualization
   * Top-K nearest frame retrieval

This pipeline transforms raw creative videos into **affective embeddings** that function as visual fingerprints.



## **Verification-First Engineering**

The system includes lightweight checks to ensure correctness before interpretation:

* Embedding shape validation
* NaN value detection
* Deterministic embedding test (identical image → identical vector)
* Safe handling of HuggingFace model outputs

These steps demonstrate a research-oriented mindset focused on pipeline reliability.



## **Results**

Observed behavior from similarity analysis:

* Frames from the **same video** cluster closely in embedding space
* Frames from **different videos** show lower similarity
* Top-K retrieval demonstrates gradual transitions in visual mood/style

This confirms CLIP embeddings capture meaningful aesthetic structure and act as **creative mood/style representations**.



## **Tech Stack**

* Python
* PyTorch
* HuggingFace Transformers
* OpenCV
* NumPy
* Matplotlib
* Pandas



## **How to Run This Project (Google Colab)**

This project is implemented as a **Google Colab notebook**, so no local setup is required.

### **1️⃣ Open the Notebook**

* Open the notebook file from this repository in **Google Colab**



### **2️⃣ Install Dependencies**

Run the setup cell:

```python
!pip install transformers torch torchvision opencv-python matplotlib pandas numpy
```



### **3️⃣ Upload Videos**

When prompted:

```python
from google.colab import files
uploaded = files.upload()
```

Upload **2–3 short royalty-free art or marketing-style videos**.



### **4️⃣ Run Frame Extraction**

This step:

* Samples frames at fixed intervals
* Saves them to a `frames/` directory
* Generates a metadata CSV file



### **5️⃣ Generate Embeddings**

This stage:

* Loads CLIP (ViT-B/32)
* Encodes frames into 512-dim embeddings
* Runs verification checks


### **6️⃣ Compute Similarity**

This section:

* Computes cosine similarity matrix
* Displays heatmap
* Retrieves top-K similar frames



### **Expected Output**

You should observe:

* A similarity heatmap
* Lists of visually similar frames
* Console logs confirming verification checks passed



### **Runtime**

Approximate runtime in Colab (CPU): **3–6 minutes** depending on video length.



## **Use of AI Coding Tools**

AI coding assistants (ChatGPT/Copilot) were used to accelerate development of boilerplate components such as model loading and preprocessing. All generated code was manually reviewed, adapted, and debugged. System design, verification logic, and interpretation were implemented with human oversight.



## **Connection to GenTA Vision**

This prototype demonstrates foundational components of a GACS-style affective system:

**Creative Frames → Embeddings → Similarity → Creative Understanding**

In a production setting, creative embeddings could be linked to performance metrics (CTR, CVR, ROAS) to learn which visual “vibes” correlate with stronger outcomes.

