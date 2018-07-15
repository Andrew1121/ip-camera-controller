from flask import Flask, jsonify, render_template, request
from net_def import build_model
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/result')
# def feedback():
#     model = build_model('cnt-k2-WorldExpo-AFCN-7c-0003-LRN_v2-my-mask-step_lr-larger_loss_weight.yaml', 'model_weights_iter_0600000.h5')
#     return model

app.run(host="0.0.0.0", port=5000)
