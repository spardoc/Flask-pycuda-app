import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from pycuda_dog import process_image
from pycuda_motion_blur import process_image_motion_blur
from pycuda_mean_filter import process_image_mean_filter  # ✅ Importa el filtro de media
from PIL import Image
import numpy as np

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not allowed_file(file.filename):
            return render_template('index.html', error='Formato de imagen no válido')

        filename = secure_filename(file.filename)
        path_in = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path_in)

        # Leer imagen en numpy
        img = Image.open(path_in).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)

        # Leer modo (cpu o gpu)
        mode = request.form.get('mode', 'cpu')

        # Tipo de procesamiento
        method = request.form.get('method', 'dog')

        # Leer y validar tamaño de máscara
        opt = request.form.get('mask_option', '9')
        if opt == 'custom':
            try:
                mask_size = int(request.form['mask_size'])
                if mask_size < 1 or mask_size > 501 or mask_size % 2 == 0:
                    raise ValueError
            except (KeyError, ValueError):
                return render_template('index.html',
                    error='Tamaño personalizado inválido: impar entre 1 y 501')
        else:
            mask_size = int(opt)

        # Procesar según método seleccionado
        if method == 'motion':
            result_np, stats = process_image_motion_blur(img_np, mask_size, mode)
            out_name = f"motion_{mode}_{mask_size}.jpg"
        elif method == 'mean':
            result_np, stats = process_image_mean_filter(img_np, mask_size, mode)
            out_name = f"mean_{mode}_{mask_size}.jpg"
        else:  # default: dog
            result_np, stats = process_image(img_np, mask_size, mode)
            out_name = f"dog_{mode}_{mask_size}.jpg"

        # Guardar imagen procesada
        path_out = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        Image.fromarray(result_np).save(path_out)

        return render_template('index.html',
                               input_image=url_for('static', filename='uploads/' + filename),
                               output_image=url_for('static', filename='uploads/' + out_name),
                               stats=stats)

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', debug=False, use_reloader=False)
