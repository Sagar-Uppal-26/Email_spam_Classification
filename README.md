# 🛡️ Email Spam Classification (Custom PyTorch LSTM)

A robust Natural Language Processing project focused on detecting spam emails. Unlike projects that rely on pre-built Transformer libraries, this project features a deep learning data pipeline and a Dual-Layer Long Short-Term Memory (LSTM) recurrent neural network built entirely from scratch in PyTorch.

## 🧠 Architecture & Tech Stack
* **Deep Learning Framework:** PyTorch
* **Model Architecture:** Custom built Dual-Layer LSTM with an Embedding Layer and Sigmoid activation head.
* **Data Processing:** Regular Expressions (Regex), Custom Vocabulary Mapping.

## 🛠️ Engineering Highlights
* **From-Scratch Data Pipeline:** Engineered a complete NLP data processing pipeline without external tokenization libraries. Features include:
  * Regex-based text cleaning to strip special characters and standardize capitalization.
  * A custom dictionary-based vocabulary builder to map words to integer IDs.
  * Sequence padding to ensure uniform tensor shapes (max length of 150 tokens) for batch processing.
* **Sequential Deep Learning:** Leveraged a dual-layer LSTM architecture to capture the sequential context and long-term dependencies of words within emails, rather than just using a basic bag-of-words approach.
* **Custom Inference Engine:** Developed an inference script that automatically processes raw string inputs through the custom tokenization pipeline, converts them to PyTorch tensors, manages GPU mapping, and outputs a confidence percentage.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Sagar-Uppal-26/Email-Spam-Classification-LSTM.git](https://github.com/Sagar-Uppal-26/Email-Spam-Classification-LSTM.git)
   cd Email-Spam-Classification-LSTM
2. Install dependencies:
    ```bash
    pip install torch pandas numpy scikit-learn
3. Run the inference script to test raw email strings against the trained LSTM weights.
