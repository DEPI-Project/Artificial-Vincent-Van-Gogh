from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import Dash

# إنشاء التطبيق
app = Dash()

# تصميم واجهة المستخدم مع تحسينات
app.layout = html.Div(id='page-content', style={
    'padding': '10px',  # تقليل الحواف
    'height': '100vh',  # استخدام ارتفاع الشاشة بالكامل
    'display': 'flex', 
    'flexDirection': 'column',
    'justifyContent': 'space-evenly',  # توزيع المحتوى عمودياً بالتساوي
    'alignItems': 'center',  # تمركز المحتوى أفقياً
    'backgroundColor': '#f8f9fa'
}, children=[
    # العنوان
    html.H1("Artificial Vincent Van Gogh", id='title', style={
        'textAlign': 'center', 
        'fontFamily': 'Arial', 
        'color': '#343a40',  
        'fontSize': '28px',  # تقليل حجم النص
        'marginBottom': '10px'  # تقليل المسافة السفلية
    }),

    # مكان عرض الصور
    html.Div(id='output-images', children=[
        html.Div([
            html.Img(src='placeholder.jpg', style={
                'width': '90%',  # تقليل عرض الصورة لتناسب الشاشة
                'maxWidth': '400px',  # أقصى عرض للصورة
                'padding': '5px',  # تقليل التباعد
                'border': '2px solid #343a40', 
                'borderRadius': '8px'
            }),
            html.A("Download Image", href='placeholder.jpg', download='vincent_image1.jpg', style={
                'display': 'block', 
                'textAlign': 'center', 
                'marginTop': '5px', 
                'color': '#007bff',
                'textDecoration': 'none',
                'fontFamily': 'Arial',
                'fontSize': '14px'
            })
        ], style={'textAlign': 'center'}),

        html.Div([
            html.Img(src='placeholder.jpg', style={
                'width': '90%',  
                'maxWidth': '400px',  # أقصى عرض للصورة
                'padding': '5px',  
                'border': '2px solid #343a40', 
                'borderRadius': '8px'
            }),
            html.A("Download Image", href='placeholder.jpg', download='vincent_image2.jpg', style={
                'display': 'block', 
                'textAlign': 'center', 
                'marginTop': '5px', 
                'color': '#007bff',
                'textDecoration': 'none',
                'fontFamily': 'Arial',
                'fontSize': '14px'
            })
        ], style={'textAlign': 'center'}),
    ], style={
        'display': 'flex',  
        'justifyContent': 'space-around',  # توزيع الصور بالتساوي
        'alignItems': 'center',  # محاذاة العناصر عمودياً في المركز
        'width': '100%',  # جعل العرض كاملاً ليتناسب مع الشاشة
    }),

    # وصف بسيط عن AI
    html.P("Generate beautiful paintings in the style of Vincent van Gogh using AI.", id='description', style={
        'textAlign': 'center', 
        'color': '#343a40',  
        'fontFamily': 'Arial', 
        'fontSize': '16px',  
        'margin': '10px 0'  # تعديل الهوامش لتقليل المساحة المستخدمة
    }),

    # الأزرار
    html.Div(children=[
        html.Button('🎨 Generate Images', id='generate-btn', style={
            'fontSize': '16px',  # حجم خط متوسط للأزرار
            'padding': '10px 20px',  # تعديل الحواف الداخلية
            'border': 'none', 
            'borderRadius': '20px',  
            'background': 'linear-gradient(135deg, #6a11cb, #2575fc)',  
            'color': 'white',
            'cursor': 'pointer',
            'marginRight': '10px',  # المسافة بين الأزرار
            'transition': '0.3s',  
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
        }),

        html.Button('💡 Toggle Light/Dark Mode', id='toggle-mode-btn', style={
            'fontSize': '16px',  
            'padding': '10px 20px',  
            'border': 'none', 
            'borderRadius': '20px',  
            'background': 'linear-gradient(135deg, #ff758c, #ff7eb3)',  
            'color': 'white',
            'cursor': 'pointer',
            'transition': '0.3s',  
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
        })
    ], style={
        'display': 'flex', 
        'justifyContent': 'center', 
        'alignItems': 'center',
        'paddingBottom': '10px'
    })
])

# وظيفة الاستجابة لتبديل الإضاءة
@app.callback(
    [Output('page-content', 'style'),
     Output('title', 'style'),
     Output('generate-btn', 'style'),
     Output('toggle-mode-btn', 'style'),
     Output('description', 'style')],  
    [Input('toggle-mode-btn', 'n_clicks')],
    [State('page-content', 'style'),
     State('title', 'style'),
     State('generate-btn', 'style'),
     State('toggle-mode-btn', 'style'),
     State('description', 'style')]  
)
def toggle_mode(n_clicks, page_style, title_style, generate_btn_style, toggle_btn_style, description_style):
    if n_clicks and n_clicks % 2 == 1:
        # الوضع الداكن
        page_style['backgroundColor'] = '#343a40'
        title_style['color'] = '#f8f9fa'
        generate_btn_style['background'] = 'linear-gradient(135deg, #495057, #6c757d)'  
        toggle_btn_style['background'] = 'linear-gradient(135deg, #495057, #6c757d)'  
        description_style['color'] = '#f8f9fa'  
    else:
        # الوضع الفاتح
        page_style['backgroundColor'] = '#f8f9fa'
        title_style['color'] = '#343a40'
        generate_btn_style['background'] = 'linear-gradient(135deg, #6a11cb, #2575fc)'  
        toggle_btn_style['background'] = 'linear-gradient(135deg, #ff758c, #ff7eb3)'  
        description_style['color'] = '#343a40'  
    
    return page_style, title_style, generate_btn_style, toggle_btn_style, description_style

# تشغيل التطبيق
if __name__ == '__main__':
    app.run_server(debug=True)
