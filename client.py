"""
This demo records sound, sends it to a server for processing, and then uses
the resulting color to either a local lighting setup or external lighting
setup
"""
import argparse
import pyaudio
import struct
import requests
import io
import datetime
import time


def main(seconds, sampling_rate, ai_service, requesttype, tableservice):
    buffer_size = sampling_rate * seconds
    print(f'buffer_size: {buffer_size}')
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sampling_rate,
        input=True,
        frames_per_buffer=buffer_size
    )

    while True:
        data_bytes = stream.read(buffer_size)
        data_stream = io.BytesIO(data_bytes)

        request_headers = {}
        request_files = {}
        request_data = None
        request_url = ai_service
        if requesttype == 'file':
            # key is defined by the server schema
            request_files = { 'audioSample': data_stream } 
            request_url += '/get_mood_color_from_audio_file'
        elif requesttype == 'stream':
            request_headers = { 'Content-Type': 'application/octet-stream' }
            request_data = data_stream
            request_url += '/get_mood_color_from_audio_stream'
        print(f'Sending request type: {requesttype} @ {datetime.datetime.now()}')
        ai_response = requests.post(
            ai_service,
            headers=request_headers,
            files=request_files,
            data=request_data
        )
        if ai_response.status_code != requests.codes.ok:
            print(f'Error in sending AI request - code: {ai_response.status_code}')
            print(ai_response.text)
        else:
            print(f'Recieved response @ {datetime.datetime.now()}')
            rgb = ai_response.json()['result']
            wrapper = { 'data': { 'data' : led_info } }
            table_response = requests.post(tableservice, json=wrapper)
            if table_response.status_code == requests.codes.ok:
                print(f'Successfully updated color to {rgb}')
            else:
                print(
                    f'Error in table request - code: {table_response.status_code}'
                )
        time.sleep(seconds)



def parse_arguments():
    client_args = argparse.ArgumentParser(
        description="Audio recoder that sends segments to a server"
    )
    client_args.add_argument(
        '-s',
        '--seconds',
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Number of seconds recorded before sent for processing"
    )
    client_args.add_argument(
        '-sr',
        '--samplingrate',
        type=int,
        choices=[4000, 8000, 16000, 32000, 44100],
        default=8000,
        help="Recording sampling rate"
    )
    client_args.add_argument(
        '-r',
        '--requesttype',
        type=str,
        choices=['file', 'stream'],
        default='stream',
        help="Format of request sent to AI Service"
    )
    client_args.add_argument(
        '-ai',
        '--aiservice',
        type=str,
        default="https://sisyphus-mood-lighting-server.herokuapp.com",
        help="URL for AI Service"
    )
    client_args.add_argument(
        '-ta',
        '--tableservice',
        type=str,
        default="http://seniordesigntable.msoe.edu:3002/sisbot/set_led_color",
        help="Service that the RGB value should be sent to"
    )
    return client_args.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.samplingrate != 8000:
        print(
            'Warning! Segment classifiers have been trained on 8KHz samples.'
            ' Therefore results will be not optimal. '
        )
    main(
        args.seconds,
        args.samplingrate,
        args.aiservice,
        args.requesttype,
        args.tableservice
    )
