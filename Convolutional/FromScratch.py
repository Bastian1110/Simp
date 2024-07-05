import numpy as np
from PIL import Image

def fft_convolve_2d_rgb(image, kernel):
    # Extract the RGB channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    # Perform FFT convolution on each channel
    red_convolved = fft_convolve_2d(red_channel, kernel)
    green_convolved = fft_convolve_2d(green_channel, kernel)
    blue_convolved = fft_convolve_2d(blue_channel, kernel)
    
    # Combine the convolved channels
    convolved_image = np.stack([red_convolved, green_convolved, blue_convolved], axis=-1)
    
    return convolved_image

def fft_convolve_2d(image, kernel):
    # Pad the image and kernel to the next power of 2 for efficiency
    n = image.shape[0] + kernel.shape[0] - 1
    m = image.shape[1] + kernel.shape[1] - 1
    n_padded = 2**np.ceil(np.log2(n)).astype(int)
    m_padded = 2**np.ceil(np.log2(m)).astype(int)
    
    # FFT of the padded image and kernel
    image_fft = np.fft.fft2(image, s=(n_padded, m_padded))
    kernel_fft = np.fft.fft2(kernel, s=(n_padded, m_padded))
    
    # Element-wise multiplication in the frequency domain
    convolved = np.fft.ifft2(image_fft * kernel_fft)
    
    # Return the real part of the inverse FFT result
    return np.real(convolved[:image.shape[0], :image.shape[1]])

# Load image and convert to RGB
image = np.array(Image.open("foo.png").convert("RGB"))

# Define the kernel
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Perform convolution per channel
convolved = fft_convolve_2d_rgb(image, kernel)

# Clip values and convert to uint8
convolved = np.clip(convolved, 0, 255).astype(np.uint8)

# Convert to image and resize
convolved_image = Image.fromarray(convolved).resize((100, 100), Image.NEAREST)

# Save the result
convolved_image.save("convolved_fft.png")
