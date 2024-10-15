import os
import tensorflow as tf
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import base64
from io import BytesIO
from PIL import Image

# Load the generator model
def load_generator(generator_builder, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return None, 0  # No checkpoints, start from epoch 0

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

# Function to generate images using the generator model
def generate_image(generator):
    # Generate a random noise vector
    noise = tf.random.normal([1, 100])  # Adjust the noise dimension as per your model's input
    generated_image = generator(noise, training=False)
    
    # Post-process the generated image (e.g., scale back to [0, 255])
    generated_image = (generated_image * 127.5 + 127.5).numpy().astype('uint8')  # Example post-processing
    
    return generated_image

# Convert the generated image to base64 format to display in Dash
def convert_image_to_base64(image):
    # Convert the image (NumPy array) to an image file object
    img = Image.fromarray(image[0])  # Assuming image is in [batch_size, width, height, channels] format
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    
    return f"data:image/png;base64,{img_base64}"

# Create the Dash app
app = Dash()

# Generator Builder (You need to implement your generator structure here)
def generator_builder():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(100,)),  # Specify input shape here
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(28 * 28 * 3, activation='tanh'),
        tf.keras.layers.Reshape((28, 28, 3))
    ])
    return model


# Load the generator model
generator, latest_epoch = load_generator(generator_builder)

# Design the layout of the app
app.layout = html.Div(id='page-content', style={
    'padding': '10px',
    'height': '100vh',
    'display': 'flex',
    'flexDirection': 'column',
    'justifyContent': 'space-evenly',
    'alignItems': 'center',
    'backgroundColor': '#f8f9fa'
}, children=[
    # Title
    html.H1("Artificial Vincent Van Gogh", id='title', style={
        'textAlign': 'center',
        'fontFamily': 'Arial',
        'color': '#343a40',
        'fontSize': '28px',
        'marginBottom': '10px'
    }),

    # Place to display the generated image
    html.Div(id='output-images', children=[
        html.Div(id='image-container', children=[], style={'textAlign': 'center'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'alignItems': 'center',
        'width': '100%'
    }),

    # Generate button
    html.Button('🎨 Generate Image', id='generate-btn', style={
        'fontSize': '16px',
        'padding': '10px 20px',
        'border': 'none',
        'borderRadius': '20px',
        'background': 'linear-gradient(135deg, #6a11cb, #2575fc)',
        'color': 'white',
        'cursor': 'pointer',
        'transition': '0.3s',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
    })
])

# Callback to generate and display image on button click
@app.callback(
    Output('image-container', 'children'),
    [Input('generate-btn', 'n_clicks')],
    [State('image-container', 'children')]  # We don't need any state, but we can use this to track updates
)
def update_output(n_clicks, children):
    if n_clicks:
        # Generate the image
        image = generate_image(generator)
        # Convert the image to base64
        image_src = convert_image_to_base64(image)
        
        # Return the new image element
        return html.Img(src=image_src, style={
            'width': '90%',
            'maxWidth': '400px',
            'padding': '5px',
            'border': '2px solid #343a40',
            'borderRadius': '8px'
        })
    return children

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
