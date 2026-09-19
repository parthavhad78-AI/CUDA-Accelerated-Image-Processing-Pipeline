# CUDA-Accelerated Image Processing Pipeline

**GPU Specialization — Capstone Project**

## Overview

This project implements a GPU-accelerated image processing pipeline using raw CUDA C++. The pipeline batch-processes images through three sequential stages, with the computationally intensive operations executed on the GPU:

1. **RGB → Grayscale Conversion** — converts each input image from RGB to grayscale using a CUDA kernel.
2. **5×5 Gaussian Blur** — reduces image noise using a fixed, normalized 5×5 Gaussian kernel stored in CUDA `__constant__` memory.
3. **3×3 Sobel Edge Detection** — calculates horizontal and vertical image gradients, combines them into a gradient magnitude, and applies a threshold to generate an edge map.

Images are loaded on the host using `stb_image`, transferred to device memory, processed through the CUDA kernels, and then copied back to the host. The resulting edge maps are saved as PNG files using `stb_image_write`.

CUDA events are used to measure GPU execution time for each image, and the timing information is stored in a CSV file for performance analysis.

---

## Project Objectives

The main objectives of this project are:

- Apply CUDA parallel programming to a real-world image processing problem.
- Implement image processing operations using custom CUDA kernels.
- Understand CUDA grid and block organization.
- Practice GPU device memory allocation and data transfers.
- Demonstrate the use of CUDA constant memory.
- Measure GPU execution time using CUDA events.
- Process multiple images in a single execution.
- Generate and analyze image-processing results.

---

## Why This Project?

Image processing is well suited to GPU acceleration because an image contains a large number of pixels that can often be processed independently.

The project uses CUDA's thread and block execution model to process image pixels in parallel. Instead of processing only one image at a time as a simple demonstration, the application supports batch processing of an entire directory of images.

The project also demonstrates lower-level CUDA programming concepts by implementing the processing operations with custom kernels rather than relying on high-level GPU image-processing libraries.

---

## Processing Pipeline

The complete processing pipeline is:

```text
Input Image
     |
     v
RGB → Grayscale
     |
     v
5×5 Gaussian Blur
     |
     v
3×3 Sobel Edge Detection
     |
     v
Thresholding
     |
     v
Generated Edge Map