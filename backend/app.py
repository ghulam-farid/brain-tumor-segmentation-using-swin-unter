from flask import Flask, send_from_directory
from flask_cors import CORS
from config import DevelopmentConfig, ProductionConfig
from routes.upload import upload_bp


def create_app(config_name='development'):
    app = Flask(__name__)

    # Load config
    if config_name == 'production':
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(DevelopmentConfig)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(upload_bp)

    # Serve predictions
    @app.route('/predictions/<path:filename>')
    def serve_predictions(filename):
        return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)

    @app.route('/')
    def index():
        return "Brain Tumor Segmentation API - Running ✓"

    return app


if __name__ == '__main__':
    app = create_app(config_name='development')
    app.run(host='0.0.0.0', port=5000, debug=True)