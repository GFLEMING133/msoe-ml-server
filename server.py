from flask import Flask, request, jsonify
from mood_lighting_ml import get_color_from_audio

app = Flask(__name__)

@app.route('/get_mood_color_from_audio_file', methods=['POST'])
def get_mood_color_from_audio_file():
    result = get_color_from_audio(request.files['audioSample'].read())
    with open('results.txt', 'a') as o:
        o.write(result + '\n')
    return jsonify({'result': result })

@app.route('/get_mood_color_from_audio_stream', methods=['POST'])
def get_mood_color_from_audio_stream():
    result = get_color_from_audio(request.get_data())
    with open('results.txt', 'a') as o:
        o.write(result + '\n')
    return jsonify({'result': result })

if __name__ == '__main__':
    app.run(host="0.0.0.0")
