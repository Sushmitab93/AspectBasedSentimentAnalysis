"""DistilBERT embedding extraction."""

import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd


class DistilBertEmbedder:
    """Extract DistilBERT CLS token embeddings from text."""

    def __init__(self, model_name: str = 'distilbert-base-uncased', device: str = None):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract_embeddings(
        self,
        sentences,
        max_length: int = 64,
        batch_size: int = 32,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Extract 768-dim CLS token embeddings from sentences.

        Args:
            sentences: List or Series of text strings
            max_length: Max token length (longer texts truncated)
            batch_size: Batch size for processing
            verbose: Print progress

        Returns:
            Numpy array of shape (len(sentences), 768)
        """
        if isinstance(sentences, pd.Series):
            sentences = sentences.tolist()
        else:
            sentences = list(sentences)

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]

                tokens = self.tokenizer(
                    batch,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # Move tokens to device
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                outputs = self.model(**tokens)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

                if verbose and (i // batch_size) % 10 == 0:
                    print(f"  Processed {i}/{len(sentences)} sentences...")

        if verbose:
            print(f"  Processed {len(sentences)}/{len(sentences)} sentences.")

        return np.vstack(all_embeddings)
