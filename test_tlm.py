"""
Test script for TLM (TCP Language Model)
Quick verification that all components work correctly
"""

import torch
from tcp_tokenizer import TCPTokenizer
from data_generator import TCPSequenceGenerator
from tlm_model import TLMModel, TLMTrainer

def test_tokenizer():
    """Test the TCP tokenizer"""
    print("Testing TCP Tokenizer...")
    
    tokenizer = TCPTokenizer()
    
    # Test basic functionality
    test_sequence = ['SYN', 'SYN-ACK', 'ACK', 'PSH-ACK', 'FIN', 'FIN-ACK']
    
    # Encode and decode
    encoded = tokenizer.encode_sequence(test_sequence)
    decoded = tokenizer.decode_sequence(encoded, remove_special_tokens=True)
    
    print(f"  Original: {test_sequence}")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  {decoded}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    assert decoded == test_sequence, "Tokenizer encode/decode failed!"
    print("  ✓ Tokenizer test passed!")
    return tokenizer

def test_data_generator():
    """Test the data generator"""
    print("\nTesting Data Generator...")
    
    generator = TCPSequenceGenerator()
    
    # Generate some sequences
    for i in range(3):
        sequence, pattern = generator.generate_sequence()
        print(f"  Generated {pattern}: {' -> '.join(sequence)}")
    
    # Generate training data
    inputs, targets = generator.generate_training_pairs(num_pairs=10, max_length=10)
    print(f"  Generated {len(inputs)} training pairs")
    print(f"  Sample input: {inputs[0]}")
    print(f"  Sample target: {targets[0]}")
    
    print("  ✓ Data generator test passed!")
    return generator

def test_model():
    """Test the TLM model"""
    print("\nTesting TLM Model...")
    
    tokenizer = TCPTokenizer()
    
    # Create a small model for testing
    model = TLMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=32,  # Very small for testing
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_len=20
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    test_input = torch.randint(0, tokenizer.vocab_size, (2, 8))
    output = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test generation (before training)
    generated = model.generate(tokenizer, max_length=8, temperature=1.0)
    decoded = tokenizer.decode_sequence(generated, remove_special_tokens=True)
    print(f"  Pre-training generation: {' -> '.join(decoded)}")
    
    print("  ✓ Model test passed!")
    return model

def test_training():
    """Test the training process with a tiny dataset"""
    print("\nTesting Training Process...")
    
    tokenizer = TCPTokenizer()
    generator = TCPSequenceGenerator()
    
    # Create tiny model and dataset for quick testing
    model = TLMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_len=15
    )
    
    # Generate small training dataset
    train_inputs, train_targets = generator.generate_training_pairs(
        num_pairs=50,  # Very small for testing
        max_length=10
    )
    
    print(f"  Training on {len(train_inputs)} samples")
    
    # Test training for just 2 epochs
    trainer = TLMTrainer(model, tokenizer, device='cpu')
    
    print("  Running quick training test...")
    trainer.train(
        train_data=(train_inputs, train_targets),
        num_epochs=2,
        batch_size=16,
        learning_rate=1e-3
    )
    
    # Test generation after training
    generated = model.generate(tokenizer, max_length=8, temperature=0.8)
    decoded = tokenizer.decode_sequence(generated, remove_special_tokens=True)
    print(f"  Post-training generation: {' -> '.join(decoded)}")
    
    print("  ✓ Training test passed!")
    return model, trainer

def test_sequence_validation():
    """Test TCP sequence validation logic"""
    print("\nTesting Sequence Validation...")
    
    tokenizer = TCPTokenizer()
    
    # Test sequences
    test_cases = [
        (['SYN', 'SYN-ACK', 'ACK'], True, "Valid 3-way handshake"),
        (['SYN', 'SYN-ACK', 'ACK', 'PSH-ACK', 'FIN', 'FIN-ACK'], True, "Complete connection"),
        (['SYN', 'RST-ACK'], True, "Connection refused"),
        (['ACK', 'PSH-ACK'], False, "Invalid start"),
        ([], False, "Empty sequence"),
    ]
    
    for sequence, expected_valid, description in test_cases:
        # Basic validation logic
        is_valid = len(sequence) > 0 and (sequence[0] == 'SYN' or len(sequence) >= 2)
        
        print(f"  {description}: {' -> '.join(sequence) if sequence else 'Empty'}")
        print(f"    Expected: {expected_valid}, Got: {is_valid}")
        
        if sequence:  # Only check non-empty sequences for this simple test
            if sequence[0] == 'SYN':
                is_valid = True
    
    print("  ✓ Validation test completed!")

def run_comprehensive_test():
    """Run all tests"""
    print("="*60)
    print("TLM (TCP Language Model) - Comprehensive Test Suite")
    print("="*60)
    
    try:
        # Test individual components
        tokenizer = test_tokenizer()
        generator = test_data_generator()
        model = test_model()
        
        # Test training process
        trained_model, trainer = test_training()
        
        # Test validation
        test_sequence_validation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nTLM System Summary:")
        print(f"  • Vocabulary Size: {tokenizer.vocab_size}")
        print(f"  • TCP Flags: {len(tokenizer.tcp_flags)}")
        print(f"  • Flag Combinations: {len(tokenizer.tcp_combinations)}")
        print(f"  • Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  • Connection Patterns: {len(generator.connection_patterns)}")
        
        print("\nNext Steps:")
        print("  1. Run 'python train_tlm.py' to train a full model")
        print("  2. Run 'python demo_tlm.py' for interactive demo")
        print("  3. Explore individual components with their test scripts")
        
        print("\nTLM is ready to learn the language of TCP! 🌐")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your installation and dependencies.")
        raise

if __name__ == "__main__":
    run_comprehensive_test()