"""
TCP Sequence Data Generator for TLM (TCP Language Model)
Generates realistic TCP connection sequences for training
"""

import random
import json
from tcp_tokenizer import TCPTokenizer

class TCPSequenceGenerator:
    def __init__(self):
        self.tokenizer = TCPTokenizer()
        
        # Define valid TCP connection patterns
        self.connection_patterns = {
            'normal_connection': [
                'SYN',
                'SYN-ACK', 
                'ACK',
                'PSH-ACK',  # Data transfer
                'PSH-ACK',  # More data
                'FIN',
                'FIN-ACK',
                'ACK'
            ],
            
            'simple_connection': [
                'SYN',
                'SYN-ACK',
                'ACK',
                'FIN',
                'FIN-ACK'
            ],
            
            'data_heavy_connection': [
                'SYN',
                'SYN-ACK',
                'ACK',
                'PSH-ACK',
                'PSH-ACK',
                'PSH-ACK',
                'PSH-ACK',
                'FIN',
                'FIN-ACK',
                'ACK'
            ],
            
            'reset_connection': [
                'SYN',
                'SYN-ACK',
                'ACK',
                'PSH-ACK',
                'RST'
            ],
            
            'failed_connection': [
                'SYN',
                'RST-ACK'
            ],
            
            'urgent_data': [
                'SYN',
                'SYN-ACK',
                'ACK',
                'URG',
                'PSH-ACK',
                'FIN',
                'FIN-ACK'
            ]
        }
        
        # Weights for different connection types
        self.pattern_weights = {
            'normal_connection': 0.4,
            'simple_connection': 0.2,
            'data_heavy_connection': 0.15,
            'reset_connection': 0.1,
            'failed_connection': 0.1,
            'urgent_data': 0.05
        }
    
    def generate_sequence(self, pattern_name=None):
        """Generate a single TCP sequence"""
        if pattern_name is None:
            # Choose pattern based on weights
            patterns = list(self.pattern_weights.keys())
            weights = list(self.pattern_weights.values())
            pattern_name = random.choices(patterns, weights=weights)[0]
        
        base_sequence = self.connection_patterns[pattern_name].copy()
        
        # Add some randomness
        sequence = self._add_variations(base_sequence)
        
        return sequence, pattern_name
    
    def _add_variations(self, sequence):
        """Add realistic variations to the base sequence"""
        varied_sequence = sequence.copy()
        
        # Randomly add extra ACKs (common in real TCP)
        for i in range(len(varied_sequence)):
            if varied_sequence[i] in ['PSH-ACK', 'SYN-ACK'] and random.random() < 0.3:
                # Sometimes add an extra ACK after data
                varied_sequence.insert(i + 1, 'ACK')
        
        # Randomly add duplicate packets (retransmissions)
        if random.random() < 0.2:
            dup_idx = random.randint(0, len(varied_sequence) - 1)
            varied_sequence.insert(dup_idx + 1, varied_sequence[dup_idx])
        
        # Randomly add ECN flags
        if random.random() < 0.1:
            ecn_idx = random.randint(0, len(varied_sequence) - 1)
            if varied_sequence[ecn_idx] == 'ACK':
                varied_sequence[ecn_idx] = 'ECE'
        
        return varied_sequence
    
    def generate_dataset(self, num_sequences=1000, max_length=20):
        """Generate a dataset of TCP sequences"""
        sequences = []
        labels = []
        
        for _ in range(num_sequences):
            sequence, pattern = self.generate_sequence()
            
            # Tokenize the sequence
            token_ids = self.tokenizer.encode_sequence(sequence)
            
            # Pad or truncate to max_length
            token_ids = self.tokenizer.pad_sequence(token_ids, max_length)
            
            sequences.append(token_ids)
            labels.append(pattern)
        
        return sequences, labels
    
    def generate_training_pairs(self, num_pairs=1000, max_length=20):
        """Generate input-output pairs for next token prediction"""
        input_sequences = []
        target_sequences = []
        
        for _ in range(num_pairs):
            sequence, _ = self.generate_sequence()
            token_ids = self.tokenizer.encode_sequence(sequence)
            
            if len(token_ids) < 2:
                continue
            
            # Create input-target pairs for next token prediction
            for i in range(1, min(len(token_ids), max_length)):
                input_seq = token_ids[:i]
                target_seq = token_ids[1:i+1]
                
                # Pad sequences
                input_padded = self.tokenizer.pad_sequence(input_seq, max_length)
                target_padded = self.tokenizer.pad_sequence(target_seq, max_length)
                
                input_sequences.append(input_padded)
                target_sequences.append(target_padded)
        
        return input_sequences, target_sequences
    
    def save_dataset(self, filename, sequences, labels=None):
        """Save dataset to JSON file"""
        data = {
            'sequences': sequences,
            'vocab_size': self.tokenizer.vocab_size,
            'vocab': self.tokenizer.vocab,
            'token_to_id': self.tokenizer.token_to_id
        }
        
        if labels is not None:
            data['labels'] = labels
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, filename):
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data['sequences'], data.get('labels', None)
    
    def print_sample_sequences(self, num_samples=5):
        """Print sample sequences for inspection"""
        print("Sample TCP Sequences:")
        print("=" * 50)
        
        for i in range(num_samples):
            sequence, pattern = self.generate_sequence()
            print(f"\nPattern: {pattern}")
            print(f"Sequence: {' -> '.join(sequence)}")
            
            # Show tokenized version
            token_ids = self.tokenizer.encode_sequence(sequence)
            decoded = self.tokenizer.decode_sequence(token_ids)
            print(f"Tokenized: {token_ids}")
            print(f"Decoded: {' -> '.join(decoded)}")

if __name__ == "__main__":
    # Test the data generator
    generator = TCPSequenceGenerator()
    
    # Print sample sequences
    generator.print_sample_sequences(5)
    
    # Generate training dataset
    print("\nGenerating training dataset...")
    sequences, labels = generator.generate_dataset(100, max_length=15)
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Sequence length: {len(sequences[0])}")
    print(f"Sample sequence: {sequences[0]}")
    print(f"Sample label: {labels[0]}")
    
    # Generate training pairs for next token prediction
    print("\nGenerating training pairs...")
    inputs, targets = generator.generate_training_pairs(200, max_length=15)
    
    print(f"Generated {len(inputs)} training pairs")
    print(f"Sample input: {inputs[0]}")
    print(f"Sample target: {targets[0]}")
    
    # Save datasets
    generator.save_dataset("tcp_sequences.json", sequences, labels)
    generator.save_dataset("tcp_training_pairs.json", inputs, targets)
    print("\nDatasets saved to tcp_sequences.json and tcp_training_pairs.json")