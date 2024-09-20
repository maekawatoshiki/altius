import numpy as np
import torch
import torch.nn.functional as F


def nchw_to_nchwc(tensor, c):
    """
    Convert NCHW tensor to NCHWc format.

    Parameters:
    - tensor: NumPy array with shape (N, C, H, W)
    - c: Block size for the channel dimension

    Returns:
    - NCHWc tensor with shape (N, C // c, H, W, c)
    """
    N, C, H, W = tensor.shape
    pad_C = (c - C % c) if C % c != 0 else 0
    if pad_C > 0:
        tensor = np.pad(tensor, ((0, 0), (0, pad_C), (0, 0), (0, 0)), "constant")
    C_padded = C + pad_C
    return tensor.reshape(N, C_padded // c, c, H, W).transpose(0, 1, 3, 4, 2)


def nchwc_to_nchw(tensor, c):
    """
    Convert NCHWc tensor back to NCHW format.

    Parameters:
    - tensor: NumPy array with shape (N, C//c, H, W, c)
    - c: Block size for the channel dimension

    Returns:
    - NCHW tensor with shape (N, C, H, W)
    """
    N, C_block, H, W, c_dim = tensor.shape
    assert c_dim == c, "Channel block size does not match."
    tensor = tensor.transpose(0, 1, 4, 2, 3).reshape(N, C_block * c, H, W)
    return tensor


def conv2d_nchwc(input_tensor, kernel, stride=1, padding=1, c=16):
    """
    Perform 2D convolution on NCHWc tensor.

    Parameters:
    - input_tensor: NumPy array with shape (N, C//c, H, W, c)
    - kernel: NumPy array with shape (K//c, C//c, R, S, c, c)
    - stride: Stride of the convolution
    - padding: Padding on each side
    - c: Block size for the channel dimension

    Returns:
    - Output tensor in NCHWc format
    """
    N, C_block, H, W, c_dim = input_tensor.shape
    K_block, C_block_k, R, S, c_in, c_out = kernel.shape

    assert C_block == C_block_k, "Input and kernel channel blocks must match."
    assert c_in == c, "Input channel inner dimension must match block size."
    assert c_out == c, "Output channel inner dimension must match block size."

    # Output dimensions
    H_out = (H + 2 * padding - R) // stride + 1
    W_out = (W + 2 * padding - S) // stride + 1

    # Initialize output tensor
    output = np.zeros((N, K_block, H_out, W_out, c_out), dtype=input_tensor.dtype)

    # Pad input
    input_padded = np.pad(
        input_tensor,
        ((0, 0), (0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="constant",
    )

    # Perform convolution
    for n in range(N):
        for k in range(K_block):
            for h in range(H_out):
                for w in range(W_out):
                    for r in range(R):
                        for s in range(S):
                            h_in = h * stride + r
                            w_in = w * stride + s
                            # Extract slices
                            input_slice = input_padded[
                                n, :, h_in, w_in, :
                            ]  # Shape: (C_block, c_in)
                            kernel_slice = kernel[
                                k, :, r, s, :, :
                            ]  # Shape: (C_block, c_in, c_out)
                            # Perform tensordot over C_block and c_in dimensions
                            conv_result = np.tensordot(
                                input_slice, kernel_slice, axes=([0, 1], [0, 1])
                            )  # Shape: (c_out,)
                            output[n, k, h, w, :] += conv_result

    return output


def conv2d_standard_nchw(input_tensor, kernel, stride=1, padding=1):
    """
    Perform standard 2D convolution on NCHW tensor using PyTorch for comparison.

    Parameters:
    - input_tensor: NumPy array with shape (N, C, H, W)
    - kernel: NumPy array with shape (K, C, R, S)
    - stride: Stride of the convolution
    - padding: Padding on each side

    Returns:
    - Output tensor as a NumPy array with shape (N, K, H_out, W_out)
    """
    # Convert NumPy arrays to PyTorch tensors
    input_pt = torch.from_numpy(input_tensor)
    kernel_pt = torch.from_numpy(kernel)

    # Perform convolution using PyTorch
    output_pt = F.conv2d(input_pt, kernel_pt, stride=stride, padding=padding)

    # Convert back to NumPy
    return output_pt.detach().numpy()


def main():
    # Parameters
    N = 1  # Batch size
    C = 32  # Number of input channels
    K = 32  # Number of output channels
    H, W = 64, 64  # Spatial dimensions
    R, S = 3, 3  # Kernel size
    stride = 1
    padding = 1
    c = 16  # Blocking factor

    # Ensure that C and K are divisible by c
    if C % c != 0 or K % c != 0:
        raise ValueError("C and K must be divisible by the blocking factor c.")

    # Generate random input tensor
    np.random.seed(42)  # For reproducibility
    input_nchw = np.random.randn(N, C, H, W).astype(np.float32)

    # Convert to NCHWc
    input_nchwc = nchw_to_nchwc(input_nchw, c)

    # Generate random kernel
    # Kernel shape: (K//c, C//c, R, S, c, c)
    kernel_nchwc = np.random.randn(K // c, C // c, R, S, c, c).astype(np.float32)

    # Perform NCHWc convolution
    output_nchwc = conv2d_nchwc(input_nchwc, kernel_nchwc, stride, padding, c)

    # Convert NCHWc output back to NCHW
    output_nchw_from_nchwc = nchwc_to_nchw(output_nchwc, c)

    # Reshape kernel to standard format for PyTorch
    # Corrected transpose: (K//c, C//c, R, S, c, c) -> (K//c, c_out, C//c, c_in, R, S)
    # Then reshape to (K, C, R, S)
    kernel_standard = kernel_nchwc.transpose(0, 5, 1, 4, 2, 3).reshape(K, C, R, S)

    # Perform standard convolution using PyTorch
    output_nchw_standard = conv2d_standard_nchw(
        input_nchw, kernel_standard, stride, padding
    )

    # Compare the outputs
    # Since both convolutions use the same weights and input, the outputs should be identical (within numerical precision)
    if np.allclose(output_nchw_from_nchwc, output_nchw_standard, atol=1e-4):
        print("Success: NCHWc convolution matches standard convolution.")
    else:
        # Compute the maximum difference
        max_diff = np.max(np.abs(output_nchw_from_nchwc - output_nchw_standard))
        print(f"Failure: Maximum difference between convolutions is {max_diff}")
        # Optionally, raise an assertion error
        assert False, "NCHWc convolution does not match standard convolution."

    # Print shapes for verification
    print("Input shape (NCHW):", input_nchw.shape)
    print("Converted Input shape (NCHWc):", input_nchwc.shape)
    print("Kernel shape (K//c, C//c, R, S, c, c):", kernel_nchwc.shape)
    print("Reshaped Kernel shape (K, C, R, S):", kernel_standard.shape)
    print("Output shape from NCHWc convolution (NCHW):", output_nchw_from_nchwc.shape)
    print("Output shape from standard convolution (NCHW):", output_nchw_standard.shape)


if __name__ == "__main__":
    main()
