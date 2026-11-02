import os
import sys

# Ensure WebApp directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
webapp_dir = os.path.join(current_dir, 'WebApp')
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from WebApp.app import app

if __name__ == '__main__':
    print("Starting Laptop Price Predictor server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
