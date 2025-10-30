# English to Kannada Transliteration using Seq2Seq Model

A deep learning project implementing sequence-to-sequence (Seq2Seq) architecture with GRU cells for transliterating English (Latin script) text to Kannada script. This character-level model learns the mapping between Latin characters and Kannada Unicode characters.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Training Process](#training-process)
- [Results](#results)
- [Approach and Methodology](#approach-and-methodology)
- [Limitations and Future Work](#limitations-and-future-work)

## Overview

Transliteration is the process of converting text from one script to another while preserving pronunciation. This project focuses on converting English romanized text (e.g., "namaskara") to Kannada script (ನಮಸ್ಕಾರ). Unlike translation which changes meaning, transliteration maintains the phonetic representation across different writing systems.

This implementation uses a character-level Seq2Seq model with GRU (Gated Recurrent Unit) cells, which is particularly effective for handling variable-length sequences and capturing the sequential dependencies between characters.

## Architecture

### Model Components

1. **Encoder**
   - Embeds input Latin characters into dense vectors
   - Processes the sequence through multiple GRU layers
   - Generates a context vector (hidden state) representing the entire input

2. **Decoder**
   - Takes the encoder's context vector as initial state
   - Generates Kannada characters one at a time
   - Uses teacher forcing during training for faster convergence
   - Implements autoregressive generation during inference

3. **Seq2Seq Wrapper**
   - Coordinates encoder and decoder operations
   - Manages teacher forcing ratio
   - Handles batch processing efficiently

### Architecture Diagram
```
Input: "namaskara"
    ↓
[Embedding Layer] → [GRU Layers] → [Context Vector]
                                          ↓
                                    [GRU Layers] → [Linear Layer]
                                          ↓
Output: "ನಮಸ್ಕಾರ"
```

## Dataset

The model is trained on three CSV files:
- `kan_train.csv` - Training data
- `kan_valid.csv` - Validation data
- `kan_test.csv` - Testing data

Each file contains two columns:
- **Latin**: Romanized Kannada words (e.g., "namaskara")
- **Native**: Kannada Unicode text (e.g., "ನಮಸ್ಕಾರ")

### Vocabulary Statistics
- **Source (Latin) Vocabulary Size**: 30 characters
- **Target (Kannada) Vocabulary Size**: 65 characters
- Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`

## Requirements

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.62.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kannada-transliteration.git
cd kannada-transliteration
```

2. Install dependencies:
```bash
pip install torch pandas numpy tqdm
```

3. Ensure dataset files are in the project directory:
```
kan_train.csv
kan_valid.csv
kan_test.csv
```

## Usage

### Training the Model

```python
# The notebook trains automatically when run
# Training configuration is already set in the code
N_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = default Adam optimizer rate
```

### Inference Example

```python
# Load the trained model
model.load_state_dict(torch.load('best-transliteration-model.pt'))

# Transliterate a word
word = "namaskara"
output = transliterate_word(model, word, source_vocab, target_vocab, device)
print(f"Input: {word}")
print(f"Output: {output}")
```

## Model Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Vocabulary Size | 30 | Latin character set size |
| Output Vocabulary Size | 65 | Kannada character set size |
| Embedding Dimension | 256 | Dense vector size for characters |
| Hidden Dimension | 512 | GRU hidden state size |
| Number of Layers | 2 | Stacked GRU layers |
| Dropout | 0.5 | Regularization rate |
| Batch Size | 128 | Training batch size |
| Learning Rate | Adam default | Optimizer learning rate |
| Epochs | 10 | Training iterations |
| Gradient Clipping | 1.0 | Prevents exploding gradients |

### Total Parameters
**5,574,977 trainable parameters**

## Training Process

### Training Pipeline

1. **Data Preprocessing**
   - Convert characters to indices using vocabulary mappings
   - Add special tokens (`<sos>`, `<eos>`)
   - Pad sequences to uniform length within batches

2. **Forward Pass**
   - Encoder processes source sequence
   - Decoder generates target sequence
   - Teacher forcing applied with 50% probability

3. **Loss Calculation**
   - CrossEntropyLoss with padding token ignored
   - Loss computed on predicted vs actual target sequences

4. **Backpropagation**
   - Gradient computation through the network
   - Gradient clipping to prevent exploding gradients
   - Parameter updates using Adam optimizer

5. **Validation**
   - Evaluate on validation set without teacher forcing
   - Monitor validation loss for early stopping
   - Save best model based on validation performance

### Training Progress

| Epoch | Train Loss | Validation Loss |
|-------|-----------|----------------|
| 1/10  | 2.313     | 1.650          |
| 2/10  | 0.885     | 0.954          |
| 3/10  | 0.536     | 0.832          |
| 4/10  | 0.406     | 0.760          |
| 5/10  | 0.337     | 0.732          |
| 6/10  | 0.301     | 0.685          |
| 7/10  | 0.265     | 0.736          |
| 8/10  | 0.241     | 0.718          |
| 9/10  | 0.223     | 0.694          |
| 10/10 | 0.198     | 0.703          |

**Best model achieved at Epoch 6 with validation loss: 0.685**

## Results

### Sample Predictions

| Source (Latin) | Actual (Kannada) | Predicted (Kannada) | Accuracy |
|---------------|------------------|---------------------|----------|
| thodagisikolluvavaralli | ತೊಡಗಿಸಿಕೊಳ್ಳುವವರಲ್ಲಿ | ತೊಡಗಿಸಿಕೊಳ್ಳುವವರಲ್ಲಿ | ✓ Perfect |
| hassan | ಹಾಸನ್ | ಹುಸನ್ | ✗ Minor error |
| anaheim | ಅನಾಹೈಮ್ | ಅನಹಿಮ್ | ✗ Partial match |
| sangameshvara | ಸಂಗಮೇಶ್ವರ | ಸಂಗಮೇಶ್ವರ | ✓ Perfect |
| ninnalli | ನಿನ್ನಲ್ಲಿ | ನಿನ್ನಲ್ಲಿ | ✓ Perfect |
| roling | ರೋಲಿಂಗ್ | ರೊಲಿಂಗ್ | ✗ Minor error |

### Performance Analysis

The model demonstrates strong performance on:
- Common Kannada words and phonetic patterns
- Long compound words with complex character sequences
- Regular phonetic mappings

Areas for improvement:
- Foreign words (e.g., "anaheim", "hassan")
- Vowel length distinctions (ಆ vs ಅ)
- Aspirated consonants

## Approach and Methodology

### 1. Problem Formulation
The transliteration task is formulated as a sequence-to-sequence problem where:
- **Input**: Variable-length sequence of Latin characters
- **Output**: Variable-length sequence of Kannada Unicode characters
- **Objective**: Learn the character-level mapping while preserving phonetic information

### 2. Model Architecture Selection
**Why Seq2Seq with GRU?**
- **Variable Length Handling**: Both input and output sequences have variable lengths
- **Sequential Dependencies**: Character order matters in both scripts
- **Context Preservation**: GRU maintains long-term dependencies better than simple RNNs
- **Efficiency**: GRU is computationally lighter than LSTM while maintaining similar performance

### 3. Character-Level Modeling
The model operates at character level rather than word level because:
- Transliteration is fundamentally a character mapping task
- Handles out-of-vocabulary words naturally
- Captures phonetic patterns at granular level
- More flexible for compound words and morphological variations

### 4. Training Strategy

**Teacher Forcing (50% ratio)**
- During training, the model receives actual target characters as input with 50% probability
- Helps model learn faster by providing correct context
- Gradual reduction prevents over-reliance on ground truth

**Gradient Clipping**
- Prevents exploding gradients in deep recurrent networks
- Ensures training stability

**Dropout Regularization**
- Applied to both encoder and decoder (0.5 rate)
- Prevents overfitting on training data
- Improves generalization to unseen words

### 5. Vocabulary Design
- **Special Tokens**: Handle sequence boundaries and unknown characters
  - `<sos>`: Start of sequence marker
  - `<eos>`: End of sequence marker
  - `<pad>`: Padding for batch processing
  - `<unk>`: Unknown character fallback
- **Character-Level**: Captures all possible phonetic combinations

### 6. Evaluation Methodology
- Validation loss monitors generalization
- Qualitative analysis on test examples
- Best model selection based on validation performance
- Character-level accuracy assessment

## Limitations and Future Work

### Current Limitations

1. **Foreign Word Handling**
   - Struggles with non-Kannada origin words
   - Limited training data for such cases

2. **Vowel Length Ambiguity**
   - Difficulty distinguishing short vs long vowels (ಅ vs ಆ)
   - Latin script doesn't always indicate length explicitly

3. **No Attention Mechanism**
   - Basic Seq2Seq without attention
   - May miss long-range dependencies

4. **Limited Context**
   - No word-level or sentence-level context
   - Purely character-based decisions

### Future Enhancements

1. **Add Attention Mechanism**
   - Bahdanau or Luong attention
   - Better handling of long sequences
   - Interpretable alignments

2. **Transformer Architecture**
   - Replace RNN with self-attention
   - Parallel processing capabilities
   - State-of-the-art performance

3. **Pretrained Embeddings**
   - Use phonetic embeddings
   - Transfer learning from related languages

4. **Data Augmentation**
   - Synthetic data generation
   - Back-transliteration
   - Noise injection for robustness

5. **Ensemble Methods**
   - Combine multiple models
   - Voting or averaging predictions

6. **Context Integration**
   - Word-level context
   - Sentence-level disambiguation
   - Language model integration

7. **Multi-task Learning**
   - Joint training with related tasks
   - Kannada-to-Latin transliteration
   - Pronunciation prediction

