import torch

def compute_row_stats_str(input_tensor, hint=""):
    """
    Calculate row statistics of input tensor (automatically flatten leading dimensions) 
    and return formatted string with hint prefix.
    
    Args:
        input_tensor: PyTorch tensor of any dimension (last dimension as feature dimension)
        hint: String to be used as the开头 of the result (default: empty string)
        
    Returns:
        Formatted string starting with hint, containing all statistics
    """
    if input_tensor.dim() == 0:
        raise ValueError("Input tensor cannot be 0-dimensional (scalar), please use at least 1D tensor")
    
    # Flatten leading dimensions to (batch_size, feature_dim)
    if input_tensor.dim() == 1:
        flattened = input_tensor.unsqueeze(0)  # Shape: (1, feature_dim)
    else:
        batch_size = input_tensor.size()[0:-1].numel()
        feature_dim = input_tensor.size(-1)
        flattened = input_tensor.reshape(batch_size, feature_dim)
    
    # Calculate statistics
    row_means = torch.mean(flattened, dim=1)
    mean_of_means = row_means.mean().item()
    
    row_vars = torch.var(flattened, dim=1, unbiased=False)
    mean_of_vars = row_vars.mean().item()
    
    row_maxes = torch.max(flattened, dim=1).values
    mean_of_maxes = row_maxes.mean().item()
    
    abs_row_means = torch.mean(torch.abs(flattened), dim=1)
    mean_of_abs_means = abs_row_means.mean().item()
    
    # Build result string
    result_parts = []
    if hint:
        result_parts.append(hint)
    result_parts.extend([
        f"1. Mean of per-row means: {mean_of_means:.4f}",
        f"2. Mean of per-row variances: {mean_of_vars:.4f}",
        f"3. Mean of per-row maxima: {mean_of_maxes:.4f}",
        f"4. Mean of per-row absolute means: {mean_of_abs_means:.4f}"
    ])
    
    return "\n".join(result_parts)

# Test examples
if __name__ == "__main__":
    print("=== 1D Tensor Test ===")
    tensor_1d = torch.randn(8)
    print(compute_row_stats_str(tensor_1d, hint="1D tensor statistics:"))
    
    print("\n=== 3D Tensor Test ===")
    tensor_3d = torch.randn(2, 3, 4)
    print(compute_row_stats_str(tensor_3d, hint="3D tensor (2×3×4) statistics:"))