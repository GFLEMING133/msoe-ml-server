from flask import Flask, request, jsonify
from mood_lighting_ml import get_coordinate_from_audio

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'yo'


@app.route('/get_mood_coordinates_from_audio_file', methods=['POST'])
def get_mood_coordinates_from_audio_file():
    return jsonify({'result': get_coordinate_from_audio(request.files['audioSample'].read())})


@app.route('/get_mood_coordinates_from_audio_stream', methods=['POST'])
def get_mood_coordinates_from_audio_stream():
    return jsonify({'result': get_coordinate_from_audio(request.get_data())})


if __name__ == '__main__':
    app.run(host="0.0.0.0")
