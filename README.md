# 🌟 **Fine-Tune Phi-2 with QLoRA!** 🌟

Welcome to **Fine-Tune Phi-2 with QLoRA**, an innovative project that demonstrates how to fine-tune the lightweight **microsoft/phi-2** language model using **QLoRA (Quantized Low-Rank Adaptation)** on a single text document! This repository showcases an efficient fine-tuning pipeline designed for resource-constrained environments, leveraging **4-bit quantization**, **LoRA**, and **PyTorch** to adapt the model to specific data (e.g., bio.txt). Perfect for developers, researchers, or AI enthusiasts looking to customize open-source LLMs with minimal hardware requirements! 🚀

🔗 **Repository Name**: Fine-Tune Phi-2 with QLoRA  
📅 **Last Updated**: June 2025  
👨‍💻 **Author**: Muhammad Ahmad Nadeem

* * *

🚀 Key Features
---------------

*   **Efficient Fine-Tuning** ⚡: Uses QLoRA to fine-tune phi-2 with 4-bit quantization, reducing memory usage.
*   **Customizable Pipeline** 🎛️: Easily adapt the model to any text document by modifying the input file.
*   **GPU Optimization** 💻: Leverages CUDA for speed (falls back to CPU if unavailable).
*   **Open-Source Tools** 🌍: Built with HuggingFace Transformers, Datasets, and PEFT for accessibility.
*   **Interactive Output** 💬: Generates text responses post-training, like answering "Who is Bruce Wayne?".

* * *

🛠️ Tech Stack
--------------

Here’s the toolkit powering this project!

🛡️ Python

![Generated Image](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)

  
🛡️ PyTorch

![Generated Image](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)

  
🛡️ HuggingFace Transformers

![Generated Image](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

  
🛡️ Datasets

![Generated Image](https://img.shields.io/badge/Datasets-2.0%2B-green?logo=huggingface)

  
🛡️ BitsAndBytes

![Generated Image](https://img.shields.io/badge/BitsAndBytes-4bit-purple)

  
🛡️ PEFT

![Generated Image](https://img.shields.io/badge/PEFT-QLoRA-blueviolet)

  
🛡️ PyMuPDF

![Generated Image](https://img.shields.io/badge/PyMuPDF-Text%20Extraction-red)

* * *

📂 Project Structure
--------------------

This repository contains a single Jupyter notebook:

*   **fine-tune.ipynb**: The main Colab notebook with the complete fine-tuning pipeline, including installation, data processing, model training, and inference.

* * *

🖥️ How to Get Started
----------------------

Follow these steps to run the project! 🚀

### Prerequisites

*   A Google Colab account with GPU access (recommended) 🌐
*   Basic knowledge of Python and Jupyter notebooks 📓

### Step 1: Open in Google Colab

Click this link to open the notebook in Colab: [Colab](https://colab.research.google.com/drive/1RsXFmlxfhfjsgLk9t4oLxJDKR6r982MH)

### Step 2: Upload Your Document

Replace the default **/content/bio.txt** with your own text file by uploading it to the Colab environment.

### Step 3: Install Dependencies

The notebook automatically installs required packages (e.g., pymupdf, transformers, datasets) when you run the first cell.

### Step 4: Run the Notebook

Execute each cell sequentially:

1.  Install dependencies.
2.  Load and process the text data.
3.  Fine-tune the phi-2 model with QLoRA.
4.  Save the trained model and test it with a sample query!

### Step 5: Test the Model

After training, use the pipeline to generate text (e.g., "Who is Bruce Wayne?") and explore the results!

* * *

📊 How It Works
---------------

1.  **Text Extraction** 📜: Extracts text from **bio.txt** using PyMuPDF.
2.  **Chunking** ✂️: Splits the text into 300-word chunks for processing.
3.  **Tokenization** 🔢: Prepares data with HuggingFace’s AutoTokenizer.
4.  **QLoRA Fine-Tuning** 🤖: Applies 4-bit quantization and LoRA to fine-tune phi-2 efficiently.
5.  **Training** 🎯: Trains the model for 3 epochs using the Trainer API.
6.  **Inference** 💬: Generates responses using the fine-tuned model via a pipeline.

* * *

🧠 Why This Project Stands Out
------------------------------

*   **Low-Resource Friendly** 🌱: Fine-tunes a large model on minimal hardware thanks to QLoRA.
*   **Practical Application** 🛠️: Demonstrates real-world use with customizable text data.
*   **Colab-Ready** 🌐: Seamlessly runs on Google Colab with free GPU support.
*   **Educational Value** 📚: Perfect for learning fine-tuning techniques and model optimization.

* * *

🤝 Contributing
---------------

Love the project? Contribute! 🍴

1.  Fork the repository.
2.  Create a branch (git checkout -b feature/new-feature).
3.  Commit changes (git commit -m "Add new feature").
4.  Push and open a Pull Request 📬.  
    Suggestions or bugs? Open an issue! 🤗

* * *

📜 License
----------

Licensed under the Apache License. See the LICENSE file for details.

* * *

🙌 Acknowledgments
------------------

*   **HuggingFace** for Transformers and Datasets.
*   **Microsoft** for the phi-2 model.
*   **PyMuPDF** for text extraction.
*   **Google Colab** for free GPU access.

* * *

🎉 **Thank you for exploring Fine-Tune Phi-2 with QLoRA!** Dive in and unleash the power of fine-tuned LLMs! 💻