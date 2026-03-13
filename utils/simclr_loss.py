import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss.
        Args:
            z_i: Embeddings from view 1 (batch_size x embed_dim)
            z_j: Embeddings from view 2 (batch_size x embed_dim)
        Returns:
            Contrastive loss
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, embed_dim)
        similarity_matrix = torch.mm(representations, representations.T)  # (2 * batch_size, 2 * batch_size)

        # Create labels
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size - 1, labels], dim=0) # -1 because the self-similarities has been removed

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1) # Remove self-similarities and reshape to (2 * batch_size, 2 * batch_size - 1)

        # Scale by temperature
        similarity_matrix /= self.temperature

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss