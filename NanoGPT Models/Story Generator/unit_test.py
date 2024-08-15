# Unit Testing Notebook

import unittest
import torch
import os
import sentencepiece as spm
from torch.nn import functional as F
import math

# Assuming the original code has been saved in a script named `model.py`
from load_model import GPTLanguageModel, encode, decode, get_batch, estimate_loss_and_perplexity, save_checkpoint

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

class TestModelFunctions(unittest.TestCase):
    
    def setUp(self):
        # Setup before each test
        self.model = GPTLanguageModel().to('cpu')
        self.device = 'cpu'
        
        # Sample data for testing
        self.sample_text = "This is a sample story."
        self.sample_encoded = encode(self.sample_text)
        self.sample_batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])) # Dummy data
        self.sample_checkpoint_path = 'model_checkpoint.pth'
    
    def test_encode(self):
        encoded = encode("This is a test")
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)

    def test_decode(self):
        decoded = decode(self.sample_encoded)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)

    def test_get_batch(self):
        x, y = get_batch('train')
        self.assertEqual(x.shape, (512, 32))
        self.assertEqual(y.shape, (512, 32))

    def test_save_checkpoint(self):
        save_checkpoint(self.model, torch.optim.Adam(self.model.parameters()), 1, self.sample_checkpoint_path, 0.1)
        self.assertTrue(os.path.exists(self.sample_checkpoint_path))
        
    def test_model_forward(self):
        logits, loss = self.model(self.sample_batch[0], self.sample_batch[1])
        self.assertIsInstance(logits, torch.Tensor)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(logits.shape[0], self.sample_batch[0].numel())

    def test_model_generate(self):
        generated = self.model.generate(self.sample_batch[0], max_new_tokens=5)
        self.assertEqual(generated.shape[1], self.sample_batch[0].shape[1] + 5)
    
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
