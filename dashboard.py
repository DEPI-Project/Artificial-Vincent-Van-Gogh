import os
import tensorflow as tf
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import base64
from io import BytesIO
from PIL import Image
from keras import layers
import numpy as np

# Load the generator model
def residual_block(input_tensor, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Add the input to the output (skip connection)
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



def generate_image(generator, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    noise = tf.random.normal([1, 100])
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

app = Dash(__name__)

def generator_builder():
    noise = layers.Input(shape=(100,))
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

app.layout = html.Div(
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'height': '100vh',
        'background': '#f0f4f8',
        'padding': '20px',
        'textAlign': 'center'
    },
    children=[
        html.H1(
            "AI Art Generator",
            style={
                'color': '#2c3e50',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '36px',
                'marginBottom': '20px'
            }
        ),
        html.Div(
            id='output-images',
            children=[
                html.Div(
                    id='image-container',
                    style={'marginBottom': '20px'}
                )
            ],
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'width': '100%'
            }
        ),
        html.Button(
            '🎨 Generate Image',
            id='generate-btn',
            style={
                'fontSize': '18px',
                'padding': '12px 24px',
                'border': 'none',
                'borderRadius': '30px',
                'background': 'linear-gradient(135deg, #6a11cb, #2575fc)',
                'color': 'white',
                'cursor': 'pointer',
                'transition': '0.3s ease',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
            },
            n_clicks=0
        ),
        dcc.Loading(
            id="loading-icon",
            type="circle",
            children=html.Div(id="loading-output")
        )
    ]
)

@app.callback(
    Output('image-container', 'children'),
    [Input('generate-btn', 'n_clicks')]
)
def update_output(n_clicks):
    if n_clicks:
        image = generate_image(generator, seed=np.random.randint(0, 10000))
        image_src = convert_image_to_base64(image)
        
        return html.Img(
            src=image_src,
            style={
                'width': '100%',
                'maxWidth': '500px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
                'marginTop': '20px'
            }
        )
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
