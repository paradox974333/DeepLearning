# Transliteration Model: Latin to Kannada Script Conversion

## Overview

This project implements a character-level sequence-to-sequence (Seq2Seq) neural network for transliterating Latin script to Kannada script. The model learns to convert romanized Kannada text into native Kannada script using an encoder-decoder architecture with GRU cells.

## Model Architecture

### Core Components

1. **Encoder**: Processes the input Latin characters sequentially and compresses the information into a hidden state representation
2. **Decoder**: Takes the encoder's final state and generates Kannada characters one at a time
3. **Character Embeddings**: Converts discrete characters into dense vector representations
4. **Recurrent Layers**: Uses GRU (Gated Recurrent Unit) cells to capture sequential dependencies

### Architecture Details

- **Input Vocabulary**: 30 unique Latin characters
- **Output Vocabulary**: 65 unique Kannada characters
- **Embedding Dimensions**: 256 for both encoder and decoder
- **Hidden State Size**: 512 units
- **Network Depth**: 2 stacked GRU layers
- **Regularization**: 50% dropout applied between layers
- **Total Parameters**: ~5.57 million trainable weights

## Technical Implementation

### Data Processing

The system implements custom vocabulary classes that:
- Map characters to numerical indices
- Include special tokens: `<pad>`, `<sos>` (start of sequence), `<eos>` (end of sequence), `<unk>` (unknown)
- Handle dynamic padding for variable-length sequences

### Training Strategy

- **Batch Size**: 128 sequences per batch
- **Optimization**: Adam optimizer with gradient clipping (max norm: 1.0)
- **Loss Function**: Cross-entropy loss (ignoring padding tokens)
- **Training Duration**: 10 epochs
- **Teacher Forcing**: 50% probability during training

The model uses teacher forcing, where during training, the decoder sometimes receives the actual target character rather than its own prediction. This helps stabilize early training.

### Inference Process

During evaluation and prediction:
1. Encoder processes the entire Latin input sequence
2. Decoder starts with `<sos>` token
3. Each predicted character feeds into the next decoding step
4. Generation stops when `<eos>` token is produced or maximum length is reached

## Dataset Requirements

The code expects three CSV files:
- `kan_train.csv`: Training data
- `kan_valid.csv`: Validation data
- `kan_test.csv`: Test data

Each file should contain two columns:
1. Latin script representation
2. Corresponding Kannada script

## Dependencies

```
torch >= 1.0
pandas
numpy
tqdm
```

## Training Results

The model demonstrates progressive improvement across epochs:
- **Epoch 1**: Training Loss: 2.313 | Validation Loss: 1.650
- **Epoch 10**: Training Loss: 0.198 | Validation Loss: 0.703

Final performance shows strong convergence with the model learning the character-level mappings effectively.

## Sample Predictions

The model successfully transliterates various Kannada words, showing high accuracy on common patterns:

| Latin Input | Target | Prediction | Match |
|------------|---------|------------|-------|
| sangameshvara | ಸಂಗಮೇಶ್ವರ | ಸಂಗಮೇಶ್ವರ | ✓ |
| ninnalli | ನಿನ್ನಲ್ಲಿ | ನಿನ್ನಲ್ಲಿ | ✓ |
| hassan | ಹಾಸನ್ | ಹುಸನ್ | ✗ |

## Usage

```python
# Load trained model
model.load_state_dict(torch.load('best-transliteration-model.pt'))

# Transliterate a word
latin_word = "namaskara"
kannada_output = transliterate_word(model, latin_word, 
                                    source_vocab, target_vocab, device)
print(f"Latin: {latin_word} → Kannada: {kannada_output}")
```

## Model Persistence

The best performing model (based on validation loss) is automatically saved to `best-transliteration-model.pt` during training.

## Hardware Acceleration

The implementation automatically detects and utilizes CUDA-enabled GPUs when available, falling back to CPU computation otherwise.

## Reproducibility

Random seeds are set for:
- Python's random module
- NumPy
- PyTorch (CPU and CUDA)
- cuDNN operations

This ensures consistent results across multiple runs.

## Future Enhancements

Potential improvements could include:
- Attention mechanism for better long-sequence handling
- Beam search decoding for multiple candidate outputs
- Bidirectional encoder for richer context
- Subword tokenization for handling rare character combinations
- Multi-task learning with additional Indic scripts



