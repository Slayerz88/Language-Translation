import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import math
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration (should match your training config)
CONFIG = {
    'max_len': 50,
    'd_model': 384,
    'nhead': 8,
    'num_layers': 6,
    'dff': 1028,
    'dropout': 0.2,
    'attention_dropout': 0.15,
}

# =====================================================
# ADVANCED TRANSFORMER MODEL (Same as in training)
# =====================================================
class AdvancedTransformerModel(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model=512, nhead=8, 
                 num_layers=6, dff=2048, max_len=50, dropout=0.2, attention_dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Enhanced embeddings with better initialization
        self.src_emb = nn.Embedding(input_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(target_vocab_size, d_model)
        
        # Learnable positional embeddings
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.pos_decoder = nn.Embedding(max_len, d_model)
        
        # Advanced transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Enhanced regularization
        self.dropout_emb = nn.Dropout(dropout)
        self.dropout_attention = nn.Dropout(attention_dropout)
        self.layer_norm_src = nn.LayerNorm(d_model)
        self.layer_norm_tgt = nn.LayerNorm(d_model)
        self.layer_norm_final = nn.LayerNorm(d_model)
        
        # Multi-layer output projection for better representations
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_vocab_size)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Enhanced weight initialization"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.1)
                elif 'weight' in name:
                    if 'layer_norm' in name or 'norm' in name:
                        nn.init.ones_(param)
                    else:
                        nn.init.xavier_uniform_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask"""
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, src, tgt_in):
        batch_size, src_len = src.shape
        _, tgt_len = tgt_in.shape
        device = src.device

        # Enhanced position embeddings
        src_pos = torch.arange(src_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(0).repeat(batch_size, 1)

        # Embeddings with proper scaling and layer norm
        src_emb = self.src_emb(src) * math.sqrt(self.d_model) + self.pos_encoder(src_pos)
        src_emb = self.layer_norm_src(self.dropout_emb(src_emb))
        
        tgt_emb = self.tgt_emb(tgt_in) * math.sqrt(self.d_model) + self.pos_decoder(tgt_pos)
        tgt_emb = self.layer_norm_tgt(self.dropout_emb(tgt_emb))

        # Create masks
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt_in == 0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)

        # Encoder-Decoder forward pass
        memory = self.transformer_encoder(
            src_emb, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Enhanced memory processing
        memory = self.dropout_attention(memory)
        
        decoder_output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Advanced output projection
        decoder_output = self.layer_norm_final(decoder_output)
        return self.output_projection(decoder_output)

# =====================================================
# TOKENIZER RECREATION FUNCTION
# =====================================================
def create_tokenizers_from_data(data_path="D:\\language\\english_spanish_data.csv", max_len=50):
    """
    Recreate tokenizers from the original data (same process as training)
    """
    print("🔧 Creating tokenizers from original data...")
    
    try:
        # Load data
        data = pd.read_csv(data_path)
        print(f"✅ Data loaded: {data.shape[0]} samples")
        
        # Process texts (same as training)
        english_texts = data['english'].astype(str).tolist()
        spanish_texts = ["<start> " + t + " <end>" for t in data['spanish'].astype(str).tolist()]
        
        # Enhanced cleaning (same function as training)
        def enhanced_cleaning(english_texts, spanish_texts, min_len=2, max_len=40):
            cleaned_en, cleaned_es = [], []
            
            for en, es in zip(english_texts, spanish_texts):
                en_clean = en.strip()
                es_clean = es.strip()
                
                # Length filtering
                en_words = len(en_clean.split())
                es_clean_words = es_clean.replace('<start>', '').replace('<end>', '').strip()
                es_words = len(es_clean_words.split())
                
                # Quality checks
                if (min_len <= en_words <= max_len and 
                    min_len <= es_words <= max_len and
                    len(en_clean) > 0 and len(es_clean) > 0 and
                    not any(char.isdigit() for char in en_clean[:20]) and
                    len(set(en_clean.split())) > 1):
                    cleaned_en.append(en_clean)
                    cleaned_es.append(es_clean)
            
            return cleaned_en, cleaned_es
        
        # Clean data
        english_texts, spanish_texts = enhanced_cleaning(english_texts, spanish_texts)
        print(f"✅ After cleaning: {len(english_texts)} samples")
        
        # Create tokenizers (same as training)
        custom_filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        
        eng_tokenizer = Tokenizer(filters=custom_filters, oov_token='<unk>', lower=True)
        eng_tokenizer.fit_on_texts(english_texts)
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        
        spa_tokenizer = Tokenizer(filters=custom_filters, oov_token='<unk>', lower=True)
        spa_tokenizer.fit_on_texts(spanish_texts)
        spa_vocab_size = len(spa_tokenizer.word_index) + 1
        
        print(f"✅ Tokenizers created:")
        print(f"   English vocab: {eng_vocab_size}")
        print(f"   Spanish vocab: {spa_vocab_size}")
        
        return eng_tokenizer, spa_tokenizer, eng_vocab_size, spa_vocab_size
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_path}")
        print("Please update the data_path in the script to point to your CSV file.")
        raise
    except Exception as e:
        print(f"❌ Error creating tokenizers: {e}")
        raise

# =====================================================
# TRANSLATOR CLASS
# =====================================================
class EnglishToSpanishTranslator:
    def __init__(self, model_path="best_advanced_transformer.pt", 
                 tokenizer_path="advanced_tokenizers.pkl",
                 data_path="D:\\language\\english_spanish_data.csv"):
        """
        Initialize the translator by loading the trained model and tokenizers
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading translator on device: {self.device}")
        
        # Try to load existing tokenizers first
        tokenizers_loaded = False
        try:
            with open(tokenizer_path, "rb") as f:
                tokenizer_data = pickle.load(f)
                self.eng_tokenizer = tokenizer_data['eng_tokenizer']
                self.spa_tokenizer = tokenizer_data['spa_tokenizer']
                self.config = tokenizer_data.get('config', CONFIG)
                self.vocab_sizes = tokenizer_data['vocab_sizes']
                tokenizers_loaded = True
                print("✅ Tokenizers loaded from file!")
                
        except FileNotFoundError:
            print("⚠️  Tokenizers file not found. Creating from data...")
            
        # If tokenizers not loaded, create them from data
        if not tokenizers_loaded:
            try:
                self.eng_tokenizer, self.spa_tokenizer, eng_vocab_size, spa_vocab_size = create_tokenizers_from_data(data_path)
                self.vocab_sizes = {'eng': eng_vocab_size, 'spa': spa_vocab_size}
                self.config = CONFIG
                
                # Save tokenizers for future use
                tokenizer_data = {
                    'eng_tokenizer': self.eng_tokenizer,
                    'spa_tokenizer': self.spa_tokenizer,
                    'config': self.config,
                    'vocab_sizes': self.vocab_sizes
                }
                
                with open(tokenizer_path, "wb") as f:
                    pickle.dump(tokenizer_data, f)
                print(f"💾 Tokenizers saved to {tokenizer_path}")
                
            except Exception as e:
                print(f"❌ Error creating tokenizers: {e}")
                print("\n🔧 Manual Setup Instructions:")
                print("1. Make sure your CSV file path is correct")
                print("2. Update the data_path parameter in the script")
                print("3. Ensure the CSV has 'english' and 'spanish' columns")
                raise
        
        # Load model
        try:
            print("📦 Loading model...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get model config from checkpoint or use defaults
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                model_config = self.config
            
            # Initialize model with same architecture
            self.model = AdvancedTransformerModel(
                input_vocab_size=self.vocab_sizes['eng'],
                target_vocab_size=self.vocab_sizes['spa'],
                d_model=model_config.get('d_model', 384),
                nhead=model_config.get('nhead', 8),
                num_layers=model_config.get('num_layers', 6),
                dff=model_config.get('dff', 1028),
                max_len=model_config.get('max_len', 50),
                dropout=model_config.get('dropout', 0.2),
                attention_dropout=model_config.get('attention_dropout', 0.15)
            ).to(self.device)
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Get model performance info
            self.best_accuracy = checkpoint.get('val_acc', 0.0)
            model_params = sum(p.numel() for p in self.model.parameters())
            
            print("✅ Model loaded successfully!")
            print(f"   Model accuracy: {self.best_accuracy*100:.2f}%")
            print(f"   Model parameters: {model_params:,}")
            print(f"   Training epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"   English vocab: {self.vocab_sizes['eng']}")
            print(f"   Spanish vocab: {self.vocab_sizes['spa']}")
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find model at {model_path}")
            print("Make sure you have run the training script first to generate the model.")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("This might be due to model architecture mismatch.")
            print("Make sure the model was trained with the same architecture.")
            raise
        
        # Get special tokens (support both bracketed and plain variants)
        self.start_token = self.spa_tokenizer.word_index.get('<start>', self.spa_tokenizer.word_index.get('start', 1))
        self.end_token = self.spa_tokenizer.word_index.get('<end>', self.spa_tokenizer.word_index.get('end', 2))
        # Sets for robust checks
        self.start_token_ids = set(filter(None, [
            self.spa_tokenizer.word_index.get('<start>'),
            self.spa_tokenizer.word_index.get('start'),
        ])) or {self.start_token}
        self.end_token_ids = set(filter(None, [
            self.spa_tokenizer.word_index.get('<end>'),
            self.spa_tokenizer.word_index.get('end'),
        ])) or {self.end_token}
        
        # Create reverse word index for Spanish
        self.spa_index_word = {v: k for k, v in self.spa_tokenizer.word_index.items()}
        
        print("🎯 Translator ready!")
        print("-" * 50)

    def translate(self, sentence, max_length=50, beam_size=1):
        """
        Translate an English sentence to Spanish
        
        Args:
            sentence: English sentence to translate
            max_length: Maximum length of output sequence
            beam_size: Beam search size (1 = greedy search)
        
        Returns:
            Translated Spanish sentence
        """
        if beam_size > 1:
            return self.translate_with_beam_search(sentence, max_length, beam_size)
        else:
            return self.translate_greedy(sentence, max_length)
    
    def translate_greedy(self, sentence, max_length=50):
        """Greedy translation (faster)"""
        try:
            # Preprocess input
            sentence = sentence.strip().lower()
            if not sentence:
                return "Error: Empty sentence"
            
            print(f"🔤 Translating: '{sentence}'")
            
            # Tokenize input
            tokens = self.eng_tokenizer.texts_to_sequences([sentence])
            if not tokens[0]:  # Empty tokenization
                return "Error: Could not tokenize sentence (unknown words)"
            
            print(f"🔢 English tokens: {tokens[0]}")
            
            tokens = pad_sequences(tokens, maxlen=max_length, padding='post', truncating='post')
            src = torch.tensor(tokens, dtype=torch.long).to(self.device)
            
            # Start translation with <start> token
            decoder_input = torch.tensor([[self.start_token]], dtype=torch.long).to(self.device)
            
            generated_tokens = []
            
            with torch.no_grad():
                for step in range(max_length):
                    # Get model prediction
                    output = self.model(src, decoder_input)
                    
                    # Get next token (greedy)
                    next_token_logits = output[:, -1, :]
                    # Greedy next token
                    values, indices = torch.topk(next_token_logits, k=min(5, next_token_logits.shape[-1]))
                    next_token = indices[0, 0].item()
                    
                    print(f"Step {step+1}: Generated token {next_token}")
                    
                    # Check for end token
                    if next_token in self.end_token_ids or next_token == 0:
                        print("🏁 End token reached")
                        break
                    
                    # Optional: avoid immediate repeats
                    if generated_tokens and next_token == generated_tokens[-1] and indices.shape[-1] > 1:
                        next_token = indices[0, 1].item()

                    # Add token to sequence
                    generated_tokens.append(next_token)
                    decoder_input = torch.cat([
                        decoder_input,
                        torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                    ], dim=1)
            
            print(f"🔢 Spanish tokens: {generated_tokens}")
            
            # Convert tokens back to text
            words = []
            for token in generated_tokens:
                word = self.spa_index_word.get(token, '<unk>')
                if word not in ['<start>', '<end>', 'start', 'end', '<unk>']:
                    words.append(word)
                print(f"Token {token} -> '{word}'")
            
            translation = ' '.join(words)
            return translation if translation else "Error: No translation generated"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def translate_with_beam_search(self, sentence, max_length=50, beam_size=3):
        """Beam search translation (more accurate but slower)"""
        try:
            # Preprocess input
            sentence = sentence.strip().lower()
            if not sentence:
                return "Error: Empty sentence"
            
            # Tokenize input
            tokens = self.eng_tokenizer.texts_to_sequences([sentence])
            if not tokens[0]:
                return "Error: Could not tokenize sentence"
            
            tokens = pad_sequences(tokens, maxlen=max_length, padding='post', truncating='post')
            src = torch.tensor(tokens, dtype=torch.long).to(self.device)
            
            # Initialize beam search
            beams = [(torch.tensor([[self.start_token]], dtype=torch.long).to(self.device), 0.0)]
            
            with torch.no_grad():
                for step in range(max_length):
                    new_beams = []
                    
                    for decoder_input, score in beams:
                        # Get model prediction
                        output = self.model(src, decoder_input)
                        next_token_logits = output[:, -1, :]
                        
                        # Get top k tokens
                        log_probs = F.log_softmax(next_token_logits, dim=-1)
                        top_k_probs, top_k_tokens = torch.topk(log_probs, beam_size, dim=-1)
                        
                        for i in range(beam_size):
                            token = top_k_tokens[0, i].item()
                            prob = top_k_probs[0, i].item()
                            new_score = score + prob
                            
                            if token in self.end_token_ids or token == 0:
                                # Beam ended
                                new_beams.append((decoder_input, new_score))
                            else:
                                # Continue beam
                                new_decoder_input = torch.cat([
                                    decoder_input,
                                    torch.tensor([[token]], dtype=torch.long).to(self.device)
                                ], dim=1)
                                new_beams.append((new_decoder_input, new_score))
                    
                    # Keep only top beams
                    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                    
                    # Check if all beams ended
                    if all(seq.size(1) > 1 and (seq[0, -1].item() in self.end_token_ids or seq[0, -1].item() == 0) for seq, _ in beams):
                        break
            
            # Get best translation
            best_sequence, _ = max(beams, key=lambda x: x[1])
            tokens = best_sequence.squeeze().cpu().numpy()[1:]  # Remove start token
            
            # Convert to text
            words = []
            for token in tokens:
                if token in self.end_token_ids or token == 0:
                    break
                word = self.spa_index_word.get(token, '<unk>')
                if word not in ['<start>', '<end>', 'start', 'end', '<unk>']:
                    words.append(word)
            
            translation = ' '.join(words)
            return translation if translation else "Error: No translation generated"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def translate_batch(self, sentences, max_length=50):
        """Translate multiple sentences at once"""
        translations = []
        print(f"Translating {len(sentences)} sentences...")
        
        for i, sentence in enumerate(sentences, 1):
            translation = self.translate(sentence, max_length)
            translations.append(translation)
            print(f"{i:2d}. EN: {sentence}")
            print(f"    ES: {translation}")
            print()
        
        return translations
    
    def interactive_mode(self):
        """Interactive translation mode"""
        print("🔄 Interactive Translation Mode")
        print("Enter English sentences to translate (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                sentence = input("EN: ").strip()
                
                if sentence.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not sentence:
                    continue
                
                # Translate with beam search for better quality
                translation = self.translate(sentence, beam_size=1)  # Start with greedy for debugging
                print(f"ES: {translation}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

# =====================================================
# MAIN EXECUTION
# =====================================================
def main():
    """Main function to run the translator"""
    print("🌍 English to Spanish Translator")
    print("=" * 50)
    
    try:
        # Initialize translator - update data_path to your CSV location
        data_path = "D:\\language\\english_spanish_data.csv"  # Update this path!
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"❌ Data file not found at: {data_path}")
            print("\n🔧 Please update the data_path variable with the correct path to your CSV file.")
            print("The CSV should have 'english' and 'spanish' columns.")
            
            # Try to find the file in current directory
            current_dir_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
            if current_dir_csv:
                print(f"\n📁 Found CSV files in current directory: {current_dir_csv}")
                print("You might want to use one of these files.")
            
            return
        
        translator = EnglishToSpanishTranslator(data_path=data_path)
        
        # Simple test translation first
        print("\n🧪 Quick Test:")
        print("-" * 20)
        test_sentence = "hello"
        translation = translator.translate(test_sentence)
        print(f"EN: {test_sentence}")
        print(f"ES: {translation}")
        
        # Example translations
        print("\n📝 Example Translations:")
        print("-" * 30)
        
        example_sentences = [
            "Hello, how are you?",
            "I love learning languages.",
            "The weather is beautiful today.",
            "What time is it?",
            "Thank you for your help."
        ]
        
        for i, sentence in enumerate(example_sentences, 1):
            translation = translator.translate(sentence)
            print(f"{i:2d}. EN: {sentence}")
            print(f"    ES: {translation}")
            print()
        
        # Interactive mode
        print("🎯 Ready for interactive translation!")
        response = input("Enter interactive mode? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            translator.interactive_mode()
        else:
            print("✅ Translation examples completed!")
            
    except Exception as e:
        print(f"❌ Failed to initialize translator: {e}")
        print("\nTroubleshooting:")
        print("1. Update the data_path variable to point to your CSV file")
        print("2. Make sure 'best_advanced_transformer.pt' exists")
        print("3. Ensure CSV has 'english' and 'spanish' columns")
        print("4. Check that the model architecture matches the training script")

if __name__ == "__main__":
    main()