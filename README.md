Latin-to-Kannada Transliteration System using Deep Learning
A character-level sequence-to-sequence neural network implementation for transliterating Latin script to Kannada script using PyTorch. This project demonstrates the application of encoder-decoder architecture with GRU (Gated Recurrent Units) for the transliteration task.​

Overview
This system converts romanized Kannada text (written in Latin script) to native Kannada script using deep learning techniques. The model learns character-level mappings between the two writing systems, enabling accurate phonetic conversion without relying on rule-based approaches.​

Key Features
Character-level processing: Operates at the granular character level for precise transliteration

Sequence-to-sequence architecture: Implements encoder-decoder framework with attention-like mechanisms

Flexible RNN cells: Supports GRU and LSTM cell types with configurable architecture

Teacher forcing: Utilizes teacher forcing during training for faster convergence

GPU acceleration: Optimized for CUDA-enabled GPUs with fallback to CPU

Robust vocabulary handling: Custom vocabulary builder with special token support

Production-ready inference: Includes complete inference pipeline for real-world deployment

Architecture
Model Components
The system comprises three primary neural network components working in tandem:​

1. Encoder
The encoder processes the input Latin character sequence and compresses it into a fixed-dimensional hidden state representation.​

Embedding layer: Converts character indices to dense vector representations (256 dimensions)

RNN layers: 2-layer GRU with 512 hidden units per layer

Dropout regularization: 50% dropout applied between layers to prevent overfitting

Output: Compressed context vector capturing input sequence semantics

2. Decoder
The decoder generates the target Kannada character sequence one character at a time using the encoder's context.​

Embedding layer: 256-dimensional character embeddings for Kannada script

RNN layers: 2-layer GRU matching encoder architecture (512 hidden units)

Dropout regularization: 50% dropout for robustness

Output projection: Linear layer mapping hidden states to vocabulary probabilities

3. Seq2Seq Wrapper
Orchestrates the encoder-decoder interaction during training and inference.​

Teacher forcing: Randomly uses ground truth vs. predicted tokens (50% probability)

Greedy decoding: Selects highest probability token at each timestep

Dynamic sequence generation: Produces variable-length outputs based on EOS token

Architecture Specifications
text
Input Vocabulary Size: 30 characters (Latin alphabet + special tokens)
Output Vocabulary Size: 65 characters (Kannada script + special tokens)
Embedding Dimensions: 256 (both encoder and decoder)
Hidden State Dimensions: 512
Number of Layers: 2
RNN Cell Type: GRU
Total Parameters: 5,574,977 trainable parameters
Dataset Structure
The system requires three CSV files for training, validation, and testing:​

kan_train.csv: Training data pairs (Latin, Kannada)

kan_valid.csv: Validation data for hyperparameter tuning

kan_test.csv: Test set for final evaluation

Data Format
Each CSV file contains two columns without headers:​

text
latin,native
namaskara,ನಮಸ್ಕಾರ
dhanyavada,ಧನ್ಯವಾದ
kannada,ಕನ್ನಡ
Special Tokens
The vocabulary system incorporates four special tokens:​

<PAD>: Padding token for batch processing

<SOS>: Start-of-sequence marker

<EOS>: End-of-sequence marker

<UNK>: Unknown character placeholder

Installation
Prerequisites
bash
Python 3.7+
CUDA 10.2+ (for GPU acceleration)
Required Dependencies
bash
pip install torch torchvision torchaudio
pip install pandas numpy tqdm
Environment Setup
python
# Set random seeds for reproducibility
import random
import numpy as np
import torch

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
Usage
Training the Model
The training process follows these sequential steps:​

1. Data Loading and Vocabulary Building
python
# Load datasets
train_df = pd.read_csv('kan_train.csv', header=None, names=['latin', 'native'])
valid_df = pd.read_csv('kan_valid.csv', header=None, names=['latin', 'native'])
test_df = pd.read_csv('kan_test.csv', header=None, names=['latin', 'native'])

# Build vocabularies
source_vocab = Vocabulary('latin')
target_vocab = Vocabulary('kannada')

# Populate vocabularies from all data
all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
for _, row in all_df.iterrows():
    source_vocab.add_sentence(str(row['latin']))
    target_vocab.add_sentence(str(row['native']))
2. Model Initialization
python
# Define hyperparameters
INPUT_DIM = source_vocab.n_chars
OUTPUT_DIM = target_vocab.n_chars
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
CELL_TYPE = 'GRU'
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Initialize model components
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, CELL_TYPE, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, CELL_TYPE, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# Initialize weights uniformly
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)
3. Training Configuration
python
# Training setup
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
BATCH_SIZE = 128
N_EPOCHS = 10
CLIP = 1  # Gradient clipping value

# Create data loaders
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                           shuffle=True, collate_fn=collate_fn)
valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, 
                           collate_fn=collate_fn)
4. Training Loop
python
for epoch in range(N_EPOCHS):
    train_loss = train_fn(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate_fn(model, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-transliteration-model.pt')
    
    print(f'Epoch: {epoch+1:02}/{N_EPOCHS}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
Inference
After training, the model can be used for transliteration:​

python
# Load trained model
model.load_state_dict(torch.load('best-transliteration-model.pt'))

def transliterate_word(model, word, source_vocab, target_vocab, device, max_len=50):
    model.eval()
    
    # Convert input word to tensor
    src_indices = [source_vocab.char2index.get(char, source_vocab.char2index['<UNK>']) 
                   for char in word]
    src_tensor = torch.LongTensor([SOS_IDX] + src_indices + [EOS_IDX]).unsqueeze(0).to(device)
    
    # Encode input
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
    
    # Decode character by character
    trg_indices = [SOS_IDX]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
        
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    # Convert indices back to characters
    trg_chars = [target_vocab.index2char[i] for i in trg_indices]
    return "".join(trg_chars[1:-1])

# Example usage
latin_word = "namaskara"
kannada_word = transliterate_word(model, latin_word, source_vocab, target_vocab, device)
print(f"Latin: {latin_word} -> Kannada: {kannada_word}")
Training Results
The model demonstrates strong learning performance over 10 epochs:​

Loss Progression
Epoch	Training Loss	Validation Loss
1	2.313	1.650
2	0.885	0.954
3	0.536	0.832
4	0.406	0.760
5	0.337	0.732
6	0.301	0.685
7	0.265	0.736
8	0.241	0.718
9	0.223	0.694
10	0.198	0.703
Key Observations
Rapid initial convergence: Training loss drops from 2.313 to 0.885 in just 2 epochs

Best validation loss: 0.685 achieved at epoch 6

Slight overfitting: Validation loss stabilizes around 0.70 while training continues to decrease

Model selection: Best model checkpoint saved based on validation loss minimization

Example Predictions
Representative test set results demonstrating model accuracy:​

text
Source: thodagisikolluvavaralli
Actual: ತೊಡಗಿಸಿಕೊಳ್ಳುವವರಲ್ಲಿ
Predicted: ತೊಡಗಿಸಿಕೊಳ್ಳುವವರಲ್ಲಿ
✓ Perfect match

Source: sangameshvara
Actual: ಸಂಗಮೇಶ್ವರ
Predicted: ಸಂಗಮೇಶ್ವರ
✓ Perfect match

Source: hassan
Actual: ಹಾಸನ್
Predicted: ಹ್ಸನ್
✗ Minor vowel discrepancy

Source: anaheim
Actual: ಅನಾಹೈಮ್
Predicted: ಅನಹಿಮ್
✗ Vowel length variation
Implementation Details
Custom Dataset Class
The TransliterationDataset class handles data preprocessing and tensor conversion:​

python
class TransliterationDataset(Dataset):
    def __init__(self, df, source_vocab, target_vocab):
        self.df = df
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
    
    def __getitem__(self, idx):
        latin_word, native_word = self.df.iloc[idx]
        
        # Convert characters to indices with UNK fallback
        src_indices = [self.source_vocab.char2index.get(char, 
                       self.source_vocab.char2index['<UNK>']) 
                       for char in str(latin_word)]
        trg_indices = [self.target_vocab.char2index.get(char, 
                       self.target_vocab.char2index['<UNK>']) 
                       for char in str(native_word)]
        
        # Add SOS and EOS tokens
        src_tensor = torch.LongTensor([SOS_IDX] + src_indices + [EOS_IDX])
        trg_tensor = torch.LongTensor([SOS_IDX] + trg_indices + [EOS_IDX])
        
        return src_tensor, trg_tensor
Padding and Batching
Variable-length sequences are handled using PyTorch's pad_sequence function:​

python
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    # Pad sequences to equal length within batch
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
    
    return src_padded, trg_padded
Gradient Clipping
To prevent exploding gradients during training:​

python
# Clip gradients to maximum norm of 1
torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
Technical Approach
Character-Level Modeling Rationale
This implementation uses character-level processing rather than word-level or subword approaches for several compelling reasons:​

Open vocabulary: Handles any input word without out-of-vocabulary issues

Morphological flexibility: Kannada is an agglutinative language with complex word formation

Phonetic accuracy: Character-level mapping preserves phonetic nuances

Data efficiency: Requires smaller vocabularies (30 vs. thousands for word-level)

Generalization: Better handles rare words and proper nouns

Teacher Forcing Strategy
The model implements stochastic teacher forcing with 50% probability:​

During training: Randomly chooses between ground truth and model prediction for next input

Purpose: Balances fast convergence with exposure to model's own predictions

During inference: Always uses model predictions (teacher forcing ratio = 0)

Optimization Techniques
Several techniques enhance model performance:​

Adam optimizer: Adaptive learning rate for faster convergence

Dropout (0.5): Applied in both encoder and decoder for regularization

Gradient clipping: Prevents exploding gradients in RNN training

Batch processing: 128 samples per batch for efficient GPU utilization

Early stopping: Best model saved based on validation loss

Limitations and Future Work
Current Limitations
Fixed architecture: Requires retraining for different language pairs

No attention mechanism: Cannot learn alignment between long sequences

Greedy decoding: May not produce globally optimal outputs

Character errors: Occasional vowel length and diacritic mistakes

No beam search: Single-path decoding limits output diversity

Potential Improvements
Attention mechanisms: Implement Bahdanau or Luong attention for better alignment

Transformer architecture: Replace RNN with self-attention for parallelization

Beam search decoding: Explore multiple hypotheses for better results

Byte-pair encoding: Hybrid character-subword approach

Transfer learning: Pre-train on multiple Indic language pairs

Evaluation metrics: Implement BLEU, character error rate (CER), word accuracy

Data augmentation: Synthetic data generation for rare character combinations

