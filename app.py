import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from pycuda_dog import process_image
from pycuda_motion_blur import process_image_motion_blur
from pycuda_mean_filter import process_image_mean_filter
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

        # Configuración GPU
        gpu_config = {}
        if mode == 'gpu':
            try:
                gpu_config['blocks_x'] = int(request.form.get('blocks_x', 16))
                gpu_config['blocks_y'] = int(request.form.get('blocks_y', 16))
                gpu_config['threads_x'] = int(request.form.get('threads_x', 16))
                gpu_config['threads_y'] = int(request.form.get('threads_y', 16))
                
                # Validar valores
                if any(v < 1 or v > 1024 for v in gpu_config.values()):
                    raise ValueError
            except (ValueError, TypeError):
                return render_template('index.html',
                    error='Configuración GPU inválida: valores deben ser entre 1 y 1024')

        # Procesar según método seleccionado
        try:
            if method == 'motion':
                result_np, stats = process_image_motion_blur(img_np, mask_size, mode, **gpu_config)
                out_name = f"motion_{mode}_{mask_size}.jpg"
            elif method == 'mean':
                result_np, stats = process_image_mean_filter(img_np, mask_size, mode, **gpu_config)
                out_name = f"mean_{mode}_{mask_size}.jpg"
            else:  # default: dog
                result_np, stats = process_image(img_np, mask_size, mode, **gpu_config)
                out_name = f"dog_{mode}_{mask_size}.jpg"
        except Exception as e:
            return render_template('index.html',
                               error=f'Error al procesar imagen: {str(e)}')

        # Guardar imagen procesada
        path_out = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        Image.fromarray(result_np).save(path_out)

        # Añadir configuración GPU a stats si es necesario
        if mode == 'gpu':
            stats.update({
                'blocks': f"{gpu_config['blocks_x']}x{gpu_config['blocks_y']}",
                'threads': f"{gpu_config['threads_x']}x{gpu_config['threads_y']}"
            })

        return render_template('index.html',
                           input_image=url_for('static', filename='uploads/' + filename),
                           output_image=url_for('static', filename='uploads/' + out_name),
                           stats=stats)

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', debug=True, use_reloader=False)