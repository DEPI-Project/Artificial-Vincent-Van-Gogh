from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import Dash

# إنشاء التطبيق
app = Dash()

# تصميم واجهة المستخدم مع تحسينات
app.layout = html.Div(id='page-content', style={
    'padding': '20px', 
    'height': '100vh', 
    'display': 'flex', 
    'flexDirection': 'column',
    'backgroundColor': '#f8f9fa'  # لون فاتح للوضع الافتراضي
}, children=[
    # العنوان
    html.H1("Artificial Vincent Van Gogh", id='title', style={
        'textAlign': 'center', 
        'fontFamily': 'Arial', 
        'color': '#343a40',  # لون داكن للنص
        'fontSize': '36px',
        'marginBottom': '20px'
    }),

    # منطقة رفع الصورة
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ', html.A('Select an Image')
            ]),
            style={
                'width': '60%', 
                'height': '60px', 
                'lineHeight': '60px', 
                'borderWidth': '2px', 
                'borderStyle': 'dashed', 
                'borderRadius': '10px', 
                'textAlign': 'center', 
                'margin': '10px auto',
                'color': '#6c757d',  # لون نص محايد
                'backgroundColor': '#e9ecef'
            },
            multiple=False  # السماح برفع صورة واحدة فقط
        )
    ], style={'textAlign': 'center'}),

    # مكان عرض الصور على شكل شبكة
    html.Div(id='output-images', children=[
        html.Img(src='placeholder.jpg', style={'width': '30%', 'padding': '10px', 'border': '2px solid #343a40'}),
        html.Img(src='placeholder.jpg', style={'width': '30%', 'padding': '10px', 'border': '2px solid #343a40'}),
        html.Img(src='placeholder.jpg', style={'width': '30%', 'padding': '10px', 'border': '2px solid #343a40'}),
    ], style={
        'display': 'grid', 
        'gridTemplateColumns': 'repeat(3, 1fr)', 
        'gap': '10px', 
        'justifyItems': 'center', 
        'alignItems': 'center', 
        'flexGrow': '1'
    }),

    # وصف بسيط عن AI
    html.P("Generate beautiful paintings in the style of Vincent van Gogh using AI.", style={
        'textAlign': 'center', 
        'color': '#343a40', 
        'fontFamily': 'Arial', 
        'fontSize': '18px',
        'marginTop': '20px',
        'marginBottom': '20px'
    }),

    # حقل إدخال النص وزر توليد الصور وتبديل الإضاءة في الأسفل
    html.Div([
        # حقل إدخال النص
        dcc.Input(id='text-input', type='text', placeholder="Enter text for the painting...", style={
            'width': '60%', 
            'padding': '10px', 
            'fontSize': '18px', 
            'marginRight': '10px',
            'borderRadius': '5px',
            'border': '1px solid #343a40',
            'flexGrow': '1'
        }),

        # زر توليد الصور
        html.Button('🎨 Generate Images', id='generate-btn', style={
            'fontSize': '18px', 
            'padding': '10px 20px', 
            'border': 'none', 
            'borderRadius': '5px',
            'backgroundColor': '#343a40', 
            'color': 'white',
            'cursor': 'pointer',
            'marginRight': '10px',
            'transition': '0.3s',  # تأثير الحركة عند hover
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
        }),

        # زر لتبديل الإضاءة
        html.Button('💡 Toggle Light/Dark Mode', id='toggle-mode-btn', style={
            'fontSize': '18px', 
            'padding': '10px 20px', 
            'border': 'none', 
            'borderRadius': '5px',
            'backgroundColor': '#343a40', 
            'color': 'white',
            'cursor': 'pointer',
            'transition': '0.3s',  # تأثير الحركة عند hover
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
        })
    ], style={
        'display': 'flex', 
        'justifyContent': 'center', 
        'alignItems': 'center', 
        'marginTop': '20px', 
        'position': 'sticky', 
        'bottom': '20px'
    })
])

# وظيفة الاستجابة لتبديل الإضاءة
@app.callback(
    [Output('page-content', 'style'),
     Output('title', 'style'),
     Output('generate-btn', 'style'),
     Output('toggle-mode-btn', 'style')],
    [Input('toggle-mode-btn', 'n_clicks')],
    [State('page-content', 'style'),
     State('title', 'style'),
     State('generate-btn', 'style'),
     State('toggle-mode-btn', 'style')]
)
def toggle_mode(n_clicks, page_style, title_style, generate_btn_style, toggle_btn_style):
    if n_clicks and n_clicks % 2 == 1:
        # الوضع الداكن
        page_style['backgroundColor'] = '#343a40'
        title_style['color'] = '#f8f9fa'
        generate_btn_style['backgroundColor'] = '#6c757d'
        generate_btn_style['color'] = '#f8f9fa'
        toggle_btn_style['backgroundColor'] = '#6c757d'
        toggle_btn_style['color'] = '#f8f9fa'
    else:
        # الوضع الفاتح
        page_style['backgroundColor'] = '#f8f9fa'
        title_style['color'] = '#343a40'
        generate_btn_style['backgroundColor'] = '#343a40'
        generate_btn_style['color'] = '#f8f9fa'
        toggle_btn_style['backgroundColor'] = '#343a40'
        toggle_btn_style['color'] = '#f8f9fa'
    
    return page_style, title_style, generate_btn_style, toggle_btn_style

# تشغيل التطبيق
# تشغيل التطبيق
app.run_server(debug=True)
