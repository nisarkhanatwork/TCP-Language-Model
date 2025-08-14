"""
TCP Tokenizer for TLM (TCP Language Model)
Handles tokenization of TCP flags and protocol sequences
"""

class TCPTokenizer:
    def __init__(self):
        # TCP flags and special tokens
        self.tcp_flags = [
            'SYN',      # Synchronize sequence numbers
            'ACK',      # Acknowledgment field significant
            'FIN',      # No more data from sender
            'RST',      # Reset the connection
            'PSH',      # Push function
            'URG',      # Urgent pointer field significant
            'ECE',      # ECN-Echo
            'CWR',      # Congestion Window Reduced
            'NS',       # Nonce Sum
        ]
        
        # Common TCP flag combinations
        self.tcp_combinations = [
            'SYN-ACK',  # Server response to SYN
            'FIN-ACK',  # Acknowledgment of FIN
            'RST-ACK',  # Reset with acknowledgment
            'PSH-ACK',  # Push with acknowledgment
        ]
        
        # Special tokens for sequence modeling
        self.special_tokens = [
            '<START>',   # Start of sequence
            '<END>',     # End of sequence
            '<PAD>',     # Padding token
            '<UNK>',     # Unknown token
            '<SEP>',     # Separator between packets
        ]
        
        # Build vocabulary
        self.vocab = self.special_tokens + self.tcp_flags + self.tcp_combinations
        self.vocab_size = len(self.vocab)
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Special token IDs
        self.start_id = self.token_to_id['<START>']
        self.end_id = self.token_to_id['<END>']
        self.pad_id = self.token_to_id['<PAD>']
        self.unk_id = self.token_to_id['<UNK>']
        self.sep_id = self.token_to_id['<SEP>']
    
    def encode(self, sequence):
        """Convert TCP sequence to token IDs"""
        if isinstance(sequence, str):
            tokens = sequence.split()
        else:
            tokens = sequence
            
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_id)
        
        return token_ids
    
    def decode(self, token_ids):
        """Convert token IDs back to TCP sequence"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        
        return tokens
    
    def encode_sequence(self, sequence, add_special_tokens=True):
        """Encode a complete TCP sequence with special tokens"""
        token_ids = self.encode(sequence)
        
        if add_special_tokens:
            token_ids = [self.start_id] + token_ids + [self.end_id]
        
        return token_ids
    
    def decode_sequence(self, token_ids, remove_special_tokens=True):
        """Decode token IDs to TCP sequence, optionally removing special tokens"""
        tokens = self.decode(token_ids)
        
        if remove_special_tokens:
            # Remove special tokens
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        return tokens
    
    def pad_sequence(self, token_ids, max_length):
        """Pad sequence to max_length"""
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            return token_ids + [self.pad_id] * (max_length - len(token_ids))
    
    def get_vocab_info(self):
        """Return vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'tcp_flags': self.tcp_flags,
            'tcp_combinations': self.tcp_combinations,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab
        }

if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = TCPTokenizer()
    
    # Test encoding/decoding
    test_sequence = "SYN SYN-ACK ACK PSH-ACK FIN FIN-ACK"
    print(f"Original sequence: {test_sequence}")
    
    encoded = tokenizer.encode_sequence(test_sequence.split())
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode_sequence(encoded)
    print(f"Decoded: {' '.join(decoded)}")
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab}")