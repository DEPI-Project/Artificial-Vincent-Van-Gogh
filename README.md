
# Artificial Vincent Van Gogh - GANs & Neural Style Transfer

## Overview
This project combines a Deep Convolutional Generative Adversarial Network (DCGAN) with neural style transfer to create stylized images inspired by Vincent van Gogh's iconic style. By leveraging the power of GANs, our model analyzes and learns from a curated dataset of Van Gogh's paintings. It can then generate new, unique images that embody Van Gogh’s post-impressionist style, including bold colors, expressive brushstrokes, and emotive compositions.

## Project Objectives
The main objectives of this project are:

- **Generate High-Quality Images**: Develop a GAN capable of generating high-quality images that mimic the artistic style of Vincent van Gogh.
- **Understand Artistic Elements**: Capture key elements of Van Gogh’s style, such as his use of color and brushstroke patterns, and reproduce them using AI.
- **Explore AI’s Role in Creativity**: Examine the capabilities and limitations of AI in replicating and extending human creativity, contributing to discussions on AI’s role in art.

## Features
- **Data Collection**: The project utilizes a dataset of Van Gogh’s paintings gathered from multiple sources like the Van Gogh Museum and Google Arts & Culture, complemented with high-resolution scans from literature.
- **Data Preprocessing**: Standardizes images to a 64x64 RGB format and segments larger images to focus on intricate details.
- **DCGAN Model**: Custom-designed architecture with:
    - A generator using a mapping network, Van Gogh-style blocks, and skip connections.
    - A discriminator featuring progressive growing, adaptive augmentation, and a custom "Van Gogh Authentication Block" for detailed style analysis.
- **Interactive Dash Application**: Allows users to interact with the model by uploading content images, generating style images, and applying style transfer.

## Project Structure
```
.
├── dashboard.py        # Dash application for style transfer
├── GANS.ipynb          # Jupyter notebook for training the DCGAN model
├── checkpoints/        # Directory for storing model checkpoints
├── README.md           # This readme file
```

## Installation

Install additional dependencies:

- Install TensorFlow (ensure compatibility with your system).
- Install Dash for the web application:

```bash
pip install dash
```

- Download the TensorFlow Hub model used for style transfer (the URL is embedded in the Dash application).

## Training the DCGAN
The `GANS.ipynb` notebook contains the training code for the DCGAN model. Follow these steps:

1. Open `GANS.ipynb` in Jupyter Notebook:

    ```bash
    jupyter notebook GANS.ipynb
    ```

2. The notebook includes sections for:
    - Importing libraries and data preprocessing.
    - Defining the generator and discriminator models.
    - Training the DCGAN using a dataset of Van Gogh's artworks.
    - Saving model checkpoints during training.

## Discriminator and Generator Training Process

### Discriminator Training:
- Train on a batch of real images (label: 1).
- Train on pre-existing fake images (label: 0).

### Generator Training:
- Generate new images and train with labels as real (1.0), aiming to fool the Discriminator.

### Evaluation & Checkpointing:
- **Save Generated Images**: Save sample images at regular intervals.
- **Model Checkpoints**: Save model weights every 2000 epochs to allow resuming training.
- **Monitor Losses**: Track Discriminator and Generator losses throughout the process.
- **Resuming Training**: Every time we start training the model, it will continue from the last checkpoint epoch instead of starting from scratch.

### Training Duration:
- Train for 6000 epochs with a batch size of 64, allowing the model to improve over time.

## Running the Dash Application
The `dashboard.py` file contains the Dash application for generating style images and applying neural style transfer.

Run the application:

```bash
python dashboard.py
```

## Usage

- **Upload a Content Image**: Click on the 'Upload Content Image' button to upload an image you want to style.
- **Generate a Style Image**: Click the 'Generate Style Image' button to create a new style image using the DCGAN generator.
- **Apply Style Transfer**: Once both images are ready, click the 'Apply Style Transfer' button to blend the style image with the content image.
- **View and Save the Result**: The resulting stylized image will be displayed on the page. You can right-click to save the image.

## Data Sources

- **Van Gogh Museum, Amsterdam**: Provided high-resolution images of their collection.
- **Google Arts & Culture**: Access to digital scans of Van Gogh's artworks from various museums.
- **Books**: "The Complete Paintings" by Ingo F. Walther and Rainer Metzger was used for additional reference.

## Future Work
- **Model Improvements**: Explore the use of larger models or different architectures for generating higher-resolution images.
- **Extended Dataset**: Incorporate other post-impressionist artists for broader style generation.
- **User Experience**: Improve the interface for easier image adjustments and more user-defined options.

## References

- **[TensorFlow Documentation](https://www.tensorflow.org/)**: https://www.tensorflow.org/
- **[TensorFlow Hub Style Transfer Model](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)**: https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
- **[Dash Documentation](https://dash.plotly.com/)**: https://dash.plotly.com/


