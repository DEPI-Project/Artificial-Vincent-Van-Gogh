import os
import tensorflow as tf
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
from io import BytesIO
from PIL import Image
from keras import layers
import numpy as np
import tensorflow_hub as hub
import cv2
import tensorflow_hub as hub
import shutil
import sys

sys.path.append(r"D:\My Laptop\Me\Programming\Machine Learning\Courses\Microsoft Machine Learning Engineer - DEPI\Final Project\Artificial-Vincent-Van-Gogh")
from output import neural_style_transfer


# def load_image(img_path):
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_image(img, channels=3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     img = img[tf.newaxis, :]
#     return img

# def process(img):
#     # Ensure the image is a float32 tensor and normalize it to [0, 1]
#     cv2.imwrite('outputs/output.png',img)
#     img = load_image('outputs/output.png')
#     return img

# def Neural_style_tranfer(content_img, style_img):
#     model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
#     try:
#         model = hub.load(model_url)
#         print("Model loaded successfully.")
#     except ValueError as e:
#         print("Failed to load model:", e)
#     content_img = process(content_img)
#     style_img = process(style_img)
#     result = model(tf.constant(content_img), tf.constant(style_img))[0]
#     return result

# Load the generator model
def residual_block(input_tensor, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_tensor])
    x = layers.ReLU()(x)
    return x

def load_generator(generator_builder, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return None, 0

    checkpoint_files = os.listdir(checkpoint_dir)
    generator_files = [f for f in checkpoint_files if f.startswith('generator_epoch_') and f.endswith('.h5')]

    if not generator_files:
        return None, 0

    epochs = [int(f.split('_epoch_')[1].split('.h5')[0]) for f in generator_files]
    latest_epoch = max(epochs)

    generator = generator_builder()
    generator.load_weights(os.path.join(checkpoint_dir, f'generator_epoch_{latest_epoch}.h5'))

    print(f"Loaded generator model from epoch {latest_epoch}")

    return generator, latest_epoch

def generate_image(generator):
    noise = tf.random.normal([1, 128])
    generated_image = generator(noise, training=False)
    generated_image = (generated_image * 127.5 + 127.5).numpy().astype('uint8')
    return generated_image

def convert_image_to_base64(image):
    img = Image.fromarray(image[0])
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_base64}"

def decode_image(contents):
    _, content_string = contents.split(',')
    image_data = base64.b64decode(content_string)
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    return image_array

app = Dash()

def generator_builder():
    noise = layers.Input(shape=(128,))
    x = layers.Dense(512 * 4 * 4)(noise)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)
    x = layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = residual_block(x, filters=256)
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    output_image = layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(x)

    model = tf.keras.Model(inputs=noise, outputs=output_image)
    return model

generator, latest_epoch = load_generator(generator_builder)

app.layout = html.Div([
    html.H1("Artificial Vincent Van Gogh", id='title', style={'textAlign': 'center'}),

    html.Div([
        dcc.Upload(
            id='upload-content-image',
            children=html.Button('📤 Upload Content Image', style={
                'background-color': '#32CD32',
                'color': 'white',
                'border': 'none',
                'border-radius': '5px',
                'padding': '10px 20px',
                'font-size': '16px',
                'cursor': 'pointer'
            }),
            multiple=False
        ),
        html.Button(
            '🎨 Generate Style Image', 
            id='generate-style-btn', 
            style={
                'background-color': '#0047AB',
                'color': 'white',
                'border': 'none',
                'border-radius': '5px',
                'padding': '10px 20px',
                'font-size': '16px',
                'margin': '10px',
                'cursor': 'pointer'
            }
        ),
        html.Button(
            '🖌️ Apply Style Transfer', 
            id='apply-style-btn', 
            disabled=True, 
            style={
                'background-color': '#8A2BE2',
                'color': 'white',
                'border': 'none',
                'border-radius': '5px',
                'padding': '10px 20px',
                'font-size': '16px',
                'margin': '10px',
                'cursor': 'pointer'
            }
        ),
    ], style={'display': 'flex', 'justify-content': 'center', 'gap': '10px', 'flex-wrap': 'wrap'}),

    html.Div(id='uploaded-content-image', style={'textAlign': 'center'}),
    html.Div(id='generated-style-image', style={'textAlign': 'center'}),
    html.Div(id='styled-image-output', style={'textAlign': 'center'}),
    
    dcc.Store(id='content-image-data'),
    dcc.Store(id='style-image-data'),
    dcc.Store(id='style-transfer-applied', data=False)
])

@app.callback(
    Output('uploaded-content-image', 'children'),
    Output('content-image-data', 'data'),
    Input('upload-content-image', 'contents')
)
def display_content_image(contents):
    if contents:
        image_array = decode_image(contents)
        return html.Img(src=contents, style={'width': '150px'}), image_array.tolist()
    return None, None

@app.callback(
    Output('generated-style-image', 'children'),
    Output('style-image-data', 'data'),
    Input('generate-style-btn', 'n_clicks')
)
def generate_style_image(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    generated_image = generate_image(generator)
    img_base64 = convert_image_to_base64(generated_image)
    return html.Img(src=img_base64, style={'width': '150px'}), generated_image[0].tolist()

@app.callback(
    Output('apply-style-btn', 'disabled'),
    Input('content-image-data', 'data'),
    Input('style-image-data', 'data')
)
def enable_style_transfer(content_image_data, style_image_data):
    return content_image_data is None or style_image_data is None

@app.callback(
    Output('styled-image-output', 'children'),
    Output('style-transfer-applied', 'data'),
    Input('apply-style-btn', 'n_clicks'),
    State('content-image-data', 'data'),
    State('style-image-data', 'data')
)
def apply_style_transfer(n_clicks, content_image_data, style_image_data):
    if n_clicks is None:
        raise PreventUpdate

    content_img = np.array(content_image_data).astype(np.float32) / 255.0
    style_img = np.array(style_image_data).astype(np.float32) / 255.0

    print("Content Image Type:", type(content_img), "Content Image Shape:", content_img.shape)
    print("Style Image Type:", type(style_img), "Style Image Shape:", style_img.shape)

    # Apply neural style transfer
    cache_dir = './tfhub_modules_cache'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    styled_image = neural_style_transfer(content_img, style_img)

    # Convert the styled image tensor to a NumPy array
    styled_image_np = styled_image.numpy()  # Convert to NumPy array

    # Remove the batch dimension if present
    if styled_image_np.ndim == 4:  # If shape is (1, height, width, channels)
        styled_image_np = styled_image_np.squeeze(axis=0)  # Remove the first dimension

    # Ensure the image is in uint8 format
    styled_image_np = (styled_image_np * 255).astype(np.uint8)  # Scale and convert to uint8

    img_base64 = convert_image_to_base64(styled_image_np)
    return html.Img(src=img_base64, style={'width': '300px'}), True




@app.callback(
    Output('uploaded-content-image', 'style'),
    Output('generated-style-image', 'style'),
    Input('style-transfer-applied', 'data')
)
def hide_images_after_transfer(applied):
    if applied:
        # Hide the images when style transfer is applied
        return {'display': 'none'}, {'display': 'none'}
    else:
        # Show the images when style transfer is not yet applied
        return {'textAlign': 'center'}, {'textAlign': 'center'}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
