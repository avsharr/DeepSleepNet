"""
Tests for model architecture: DeepSleepNet forward pass, output shape, parameters.
"""
import sys
import os
import torch
import pytest

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import DeepSleepNet


class TestModelArchitecture:
    """Test model architecture and forward pass."""
    
    def test_model_initialization(self):
        """Test that model can be initialized."""
        model = DeepSleepNet(n_classes=5)
        assert model is not None, "Model should be initialized"
    
    def test_model_forward_pass(self):
        """Test that model forward pass works correctly."""
        model = DeepSleepNet(n_classes=5)
        model.eval()
        
        # Create dummy input: (batch_size, seq_length, n_channels, signal_length)
        # Model expects (batch, seq, 1, 3000)
        batch_size = 2
        seq_length = 25
        n_channels = 1
        signal_length = 3000
        dummy_input = torch.randn(batch_size, seq_length, n_channels, signal_length)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output should be (batch_size * seq_length, n_classes)
        expected_shape = (batch_size * seq_length, 5)
        assert output.shape == expected_shape, f"Output shape should be {expected_shape}, got {output.shape}"
    
    def test_model_output_range(self):
        """Test that model outputs are logits (not probabilities)."""
        model = DeepSleepNet(n_classes=5)
        model.eval()
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Logits can be any real number (not constrained to [0, 1])
        # But we can check that softmax would give valid probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0])), "Softmax should sum to 1"
        assert torch.all(probs >= 0), "Probabilities should be >= 0"
        assert torch.all(probs <= 1), "Probabilities should be <= 1"
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = DeepSleepNet(n_classes=5)
        
        # Count trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert num_params > 0, "Model should have trainable parameters"
        # DeepSleepNet should have millions of parameters
        assert num_params > 1_000_000, f"Model should have >1M parameters, got {num_params:,}"
    
    def test_model_device_compatibility(self):
        """Test that model can be moved to different devices."""
        model = DeepSleepNet(n_classes=5)
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        
        # Test CPU
        model_cpu = model.cpu()
        with torch.no_grad():
            output_cpu = model_cpu(dummy_input)
        assert output_cpu.device.type == 'cpu', "Model should work on CPU"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            dummy_input_cuda = dummy_input.cuda()
            with torch.no_grad():
                output_cuda = model_cuda(dummy_input_cuda)
            assert output_cuda.device.type == 'cuda', "Model should work on CUDA"
        
        # Test MPS if available (Apple Silicon)
        if torch.backends.mps.is_available():
            model_mps = model.to('mps')
            dummy_input_mps = dummy_input.to('mps')
            with torch.no_grad():
                output_mps = model_mps(dummy_input_mps)
            assert output_mps.device.type == 'mps', "Model should work on MPS"


class TestModelTrainingMode:
    """Test model behavior in train vs eval mode."""
    
    def test_model_train_mode(self):
        """Test that model can be set to training mode."""
        model = DeepSleepNet(n_classes=5)
        model.train()
        
        assert model.training, "Model should be in training mode"
    
    def test_model_eval_mode(self):
        """Test that model can be set to evaluation mode."""
        model = DeepSleepNet(n_classes=5)
        model.eval()
        
        assert not model.training, "Model should be in evaluation mode"
    
    def test_model_dropout_behavior(self):
        """Test that dropout behaves differently in train vs eval."""
        model = DeepSleepNet(n_classes=5)
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        
        # In training mode, dropout is active (outputs may vary)
        model.train()
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = model(dummy_input)
        
        # In eval mode, dropout is disabled (outputs should be identical)
        model.eval()
        with torch.no_grad():
            output3 = model(dummy_input)
            output4 = model(dummy_input)
        
        # Eval outputs should be identical
        assert torch.allclose(output3, output4), "Eval mode outputs should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
