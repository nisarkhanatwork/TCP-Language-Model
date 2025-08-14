"""
Demo script for TLM (TCP Language Model)
Interactive demonstration of the trained TCP sequence generator
"""

import torch
import os
from tcp_tokenizer import TCPTokenizer
from data_generator import TCPSequenceGenerator
from tlm_model import TLMModel, TLMTrainer

class TLMDemo:
    def __init__(self, model_path="tlm_trained.pth"):
        """Initialize the demo with a trained model"""
        self.tokenizer = TCPTokenizer()
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if available
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model file {model_path} not found. Creating and training a new model...")
            self.create_and_train_model()
    
    def create_and_train_model(self):
        """Create and train a new model if no trained model exists"""
        print("Creating new TLM model...")
        
        # Create model
        self.model = TLMModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=6,
            d_ff=512,
            max_len=50
        )
        
        # Generate training data
        generator = TCPSequenceGenerator()
        train_inputs, train_targets = generator.generate_training_pairs(
            num_pairs=1000,  # Smaller dataset for demo
            max_length=15
        )
        
        # Train the model
        trainer = TLMTrainer(self.model, self.tokenizer, device=self.device)
        print("Training model (this may take a few minutes)...")
        trainer.train(
            train_data=(train_inputs, train_targets),
            num_epochs=15,
            batch_size=32,
            learning_rate=1e-4
        )
        
        # Save the model
        trainer.save_model(self.model_path)
        print(f"Model trained and saved to {self.model_path}")
    
    def load_model(self):
        """Load a pre-trained model"""
        print(f"Loading model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with saved parameters - include all architecture parameters
        self.model = TLMModel(
            vocab_size=checkpoint.get('vocab_size', self.tokenizer.vocab_size),
            d_model=checkpoint.get('d_model', 128),
            n_heads=checkpoint.get('n_heads', 8),
            n_layers=checkpoint.get('n_layers', 6),
            d_ff=checkpoint.get('d_ff', 512),
            max_len=checkpoint.get('max_len', 50)
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_sequence(self, prompt=None, max_length=15, temperature=0.7, top_k=5):
        """Generate a TCP sequence"""
        if prompt:
            # Convert string prompt to token IDs
            if isinstance(prompt, str):
                prompt_tokens = prompt.split()
            else:
                prompt_tokens = prompt
            
            prompt_ids = [self.tokenizer.start_id]
            for token in prompt_tokens:
                if token in self.tokenizer.token_to_id:
                    prompt_ids.append(self.tokenizer.token_to_id[token])
        else:
            prompt_ids = None
        
        # Generate sequence
        generated = self.model.generate(
            self.tokenizer,
            prompt_tokens=prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode to readable format
        decoded = self.tokenizer.decode_sequence(generated, remove_special_tokens=True)
        return decoded
    
    def analyze_sequence(self, sequence):
        """Analyze a TCP sequence for protocol compliance"""
        analysis = {
            'valid': False,
            'connection_type': 'unknown',
            'phases': [],
            'issues': []
        }
        
        if not sequence:
            analysis['issues'].append("Empty sequence")
            return analysis
        
        # Identify connection phases
        if 'SYN' in sequence:
            analysis['phases'].append('Connection Initiation')
        
        if 'SYN-ACK' in sequence and 'ACK' in sequence:
            analysis['phases'].append('Three-way Handshake')
        
        if 'PSH-ACK' in sequence:
            analysis['phases'].append('Data Transfer')
        
        if 'FIN' in sequence or 'RST' in sequence:
            analysis['phases'].append('Connection Termination')
        
        # Determine connection type
        if sequence == ['SYN', 'SYN-ACK', 'ACK']:
            analysis['connection_type'] = 'Basic Handshake'
        elif 'PSH-ACK' in sequence:
            analysis['connection_type'] = 'Data Connection'
        elif 'RST' in sequence:
            analysis['connection_type'] = 'Reset Connection'
        elif 'FIN' in sequence:
            analysis['connection_type'] = 'Graceful Close'
        
        # Basic validation
        if len(sequence) >= 3 and sequence[0] == 'SYN':
            analysis['valid'] = True
        
        return analysis
    
    def interactive_demo(self):
        """Run interactive demo"""
        print("\n" + "="*60)
        print("TLM (TCP Language Model) Interactive Demo")
        print("="*60)
        print("This demo shows how TLM generates TCP protocol sequences.")
        print("TLM understands TCP flags and can generate realistic connection flows.")
        print("\nAvailable commands:")
        print("  1. Generate random sequence")
        print("  2. Complete a partial sequence")
        print("  3. Show vocabulary")
        print("  4. Batch generation")
        print("  5. Exit")
        
        while True:
            print("\n" + "-"*40)
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                self.demo_random_generation()
            elif choice == '2':
                self.demo_sequence_completion()
            elif choice == '3':
                self.show_vocabulary()
            elif choice == '4':
                self.demo_batch_generation()
            elif choice == '5':
                print("Thanks for trying TLM! Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
    
    def demo_random_generation(self):
        """Demo random sequence generation"""
        print("\nGenerating random TCP sequences...")
        
        for i in range(3):
            sequence = self.generate_sequence(temperature=0.8)
            analysis = self.analyze_sequence(sequence)
            
            print(f"\nSequence {i+1}: {' -> '.join(sequence)}")
            print(f"Connection Type: {analysis['connection_type']}")
            print(f"Phases: {', '.join(analysis['phases']) if analysis['phases'] else 'None detected'}")
            print(f"Valid: {'Yes' if analysis['valid'] else 'No'}")
    
    def demo_sequence_completion(self):
        """Demo sequence completion"""
        print("\nSequence Completion Demo")
        print("Enter TCP flags separated by spaces (e.g., 'SYN SYN-ACK')")
        print("Or press Enter for examples")
        
        user_input = input("Enter partial sequence: ").strip()
        
        if not user_input:
            # Use example prompts
            examples = [
                ['SYN'],
                ['SYN', 'SYN-ACK'],
                ['SYN', 'SYN-ACK', 'ACK']
            ]
            
            for prompt in examples:
                completed = self.generate_sequence(prompt=prompt, temperature=0.6)
                print(f"Input: {' -> '.join(prompt)}")
                print(f"Completion: {' -> '.join(completed)}")
                print()
        else:
            prompt = user_input.split()
            completed = self.generate_sequence(prompt=prompt, temperature=0.6)
            analysis = self.analyze_sequence(completed)
            
            print(f"Input: {' -> '.join(prompt)}")
            print(f"Completion: {' -> '.join(completed)}")
            print(f"Analysis: {analysis['connection_type']}")
    
    def show_vocabulary(self):
        """Show the TCP vocabulary"""
        print("\nTCP Language Model Vocabulary:")
        print(f"Total vocabulary size: {self.tokenizer.vocab_size}")
        
        print(f"\nTCP Flags ({len(self.tokenizer.tcp_flags)}):")
        for flag in self.tokenizer.tcp_flags:
            print(f"  {flag}")
        
        print(f"\nTCP Flag Combinations ({len(self.tokenizer.tcp_combinations)}):")
        for combo in self.tokenizer.tcp_combinations:
            print(f"  {combo}")
        
        print(f"\nSpecial Tokens ({len(self.tokenizer.special_tokens)}):")
        for token in self.tokenizer.special_tokens:
            print(f"  {token}")
    
    def demo_batch_generation(self):
        """Generate multiple sequences and analyze patterns"""
        print("\nGenerating batch of TCP sequences for analysis...")
        
        sequences = []
        connection_types = {}
        
        for i in range(10):
            sequence = self.generate_sequence(temperature=0.7)
            analysis = self.analyze_sequence(sequence)
            sequences.append((sequence, analysis))
            
            conn_type = analysis['connection_type']
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        print(f"\nGenerated {len(sequences)} sequences:")
        for i, (seq, analysis) in enumerate(sequences, 1):
            print(f"{i:2d}. {' -> '.join(seq)} ({analysis['connection_type']})")
        
        print(f"\nConnection Type Distribution:")
        for conn_type, count in connection_types.items():
            print(f"  {conn_type}: {count}")

def main():
    """Main demo function"""
    print("Initializing TLM Demo...")
    
    try:
        demo = TLMDemo()
        demo.interactive_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure PyTorch is installed and the model files are accessible.")

if __name__ == "__main__":
    main()