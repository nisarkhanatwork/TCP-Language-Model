"""
Training script for TLM (TCP Language Model)
Complete training pipeline with data generation, model training, and evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tcp_tokenizer import TCPTokenizer
from data_generator import TCPSequenceGenerator
from tlm_model import TLMModel, TLMTrainer

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def evaluate_model(model, tokenizer, num_samples=10):
    """Evaluate the trained model by generating sequences"""
    print("\n" + "="*60)
    print("EVALUATING TRAINED TLM MODEL")
    print("="*60)
    
    model.eval()
    
    # Test different starting prompts
    test_prompts = [
        None,  # Start from scratch
        [tokenizer.start_id, tokenizer.token_to_id['SYN']],  # Start with SYN
        [tokenizer.start_id, tokenizer.token_to_id['SYN'], tokenizer.token_to_id['SYN-ACK']],  # SYN, SYN-ACK
    ]
    
    prompt_names = ["From scratch", "Starting with SYN", "Starting with SYN, SYN-ACK"]
    
    for i, (prompt, name) in enumerate(zip(test_prompts, prompt_names)):
        print(f"\n{name}:")
        print("-" * 40)
        
        for j in range(3):  # Generate 3 sequences for each prompt
            generated = model.generate(
                tokenizer, 
                prompt_tokens=prompt, 
                max_length=15, 
                temperature=0.8,
                top_k=5
            )
            decoded = tokenizer.decode_sequence(generated, remove_special_tokens=True)
            print(f"  {j+1}: {' -> '.join(decoded)}")
    
    # Test sequence completion
    print(f"\nSequence Completion Test:")
    print("-" * 40)
    
    # Test completing a partial connection
    partial_sequences = [
        ['SYN'],
        ['SYN', 'SYN-ACK'],
        ['SYN', 'SYN-ACK', 'ACK'],
        ['SYN', 'SYN-ACK', 'ACK', 'PSH-ACK']
    ]
    
    for partial in partial_sequences:
        prompt_ids = [tokenizer.start_id] + [tokenizer.token_to_id[token] for token in partial]
        generated = model.generate(
            tokenizer,
            prompt_tokens=prompt_ids,
            max_length=12,
            temperature=0.5
        )
        decoded = tokenizer.decode_sequence(generated, remove_special_tokens=True)
        print(f"  Input: {' -> '.join(partial)}")
        print(f"  Completion: {' -> '.join(decoded)}")
        print()

def validate_tcp_sequence(sequence, tokenizer):
    """Validate if a generated sequence follows TCP protocol rules"""
    if not sequence:
        return False, "Empty sequence"
    
    # Remove special tokens
    clean_sequence = [token for token in sequence if token not in tokenizer.special_tokens]
    
    if not clean_sequence:
        return False, "No valid TCP tokens"
    
    # Basic TCP validation rules
    rules_passed = []
    
    # Rule 1: Should start with SYN (for new connections)
    if clean_sequence[0] == 'SYN':
        rules_passed.append("Starts with SYN")
    
    # Rule 2: SYN should be followed by SYN-ACK
    if len(clean_sequence) > 1 and clean_sequence[0] == 'SYN' and clean_sequence[1] == 'SYN-ACK':
        rules_passed.append("SYN followed by SYN-ACK")
    
    # Rule 3: Should contain ACK after SYN-ACK
    if 'SYN-ACK' in clean_sequence and 'ACK' in clean_sequence:
        syn_ack_idx = clean_sequence.index('SYN-ACK')
        ack_idx = clean_sequence.index('ACK')
        if ack_idx > syn_ack_idx:
            rules_passed.append("ACK after SYN-ACK")
    
    # Rule 4: Connection should end properly (FIN or RST)
    if clean_sequence[-1] in ['FIN', 'RST', 'FIN-ACK', 'RST-ACK']:
        rules_passed.append("Proper connection termination")
    
    return len(rules_passed) >= 2, rules_passed

def analyze_generated_sequences(model, tokenizer, num_sequences=50):
    """Analyze the quality of generated sequences"""
    print("\n" + "="*60)
    print("SEQUENCE QUALITY ANALYSIS")
    print("="*60)
    
    valid_sequences = 0
    rule_counts = {}
    
    for i in range(num_sequences):
        generated = model.generate(tokenizer, max_length=12, temperature=0.7)
        decoded = tokenizer.decode_sequence(generated)
        
        is_valid, rules = validate_tcp_sequence(decoded, tokenizer)
        
        if is_valid:
            valid_sequences += 1
        
        for rule in rules:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    print(f"Valid sequences: {valid_sequences}/{num_sequences} ({valid_sequences/num_sequences*100:.1f}%)")
    print("\nRule compliance:")
    for rule, count in rule_counts.items():
        print(f"  {rule}: {count}/{num_sequences} ({count/num_sequences*100:.1f}%)")

def main():
    """Main training function"""
    print("="*60)
    print("TLM (TCP Language Model) Training")
    print("="*60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize components
    print("\nInitializing components...")
    tokenizer = TCPTokenizer()
    generator = TCPSequenceGenerator()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"TCP flags: {tokenizer.tcp_flags}")
    print(f"TCP combinations: {tokenizer.tcp_combinations}")
    
    # Generate sample sequences for inspection
    print("\nSample TCP sequences:")
    generator.print_sample_sequences(3)
    
    # Generate training data
    print("\nGenerating training data...")
    train_inputs, train_targets = generator.generate_training_pairs(
        num_pairs=2000, 
        max_length=15
    )
    
    print(f"Generated {len(train_inputs)} training pairs")
    
    # Create model
    print("\nCreating TLM model...")
    model = TLMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,      # Embedding dimension
        n_heads=8,        # Number of attention heads
        n_layers=6,       # Number of transformer layers
        d_ff=512,         # Feed-forward dimension
        max_len=50        # Maximum sequence length
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Initialize trainer
    trainer = TLMTrainer(model, tokenizer, device=device)
    
    # Test model before training
    print("\nTesting model before training...")
    generated = model.generate(tokenizer, max_length=8, temperature=1.0)
    decoded = tokenizer.decode_sequence(generated)
    print(f"Pre-training generation: {' -> '.join(decoded)}")
    
    # Train the model
    print("\nStarting training...")
    trainer.train(
        train_data=(train_inputs, train_targets),
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-4
    )
    
    # Save the trained model
    print("\nSaving trained model...")
    trainer.save_model("tlm_trained.pth")
    
    # Evaluate the trained model
    evaluate_model(model, tokenizer)
    
    # Analyze sequence quality
    analyze_generated_sequences(model, tokenizer, num_sequences=100)
    
    # Generate and save some example sequences
    print("\n" + "="*60)
    print("FINAL EXAMPLE GENERATIONS")
    print("="*60)
    
    for i in range(5):
        generated = model.generate(tokenizer, max_length=12, temperature=0.6)
        decoded = tokenizer.decode_sequence(generated, remove_special_tokens=True)
        is_valid, rules = validate_tcp_sequence(decoded, tokenizer)
        
        print(f"\nExample {i+1}: {' -> '.join(decoded)}")
        print(f"Valid: {is_valid}")
        if rules:
            print(f"Rules passed: {', '.join(rules)}")
    
    print(f"\nTraining completed! Model saved as 'tlm_trained.pth'")
    print("You can now use the trained TLM model to generate TCP sequences!")

if __name__ == "__main__":
    main()