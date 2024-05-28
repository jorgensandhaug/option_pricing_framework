from flask import Flask, redirect, request
import webbrowser
import base64
import requests
import os
import sys
import string
import random
import threading

app = Flask(__name__)

CLIENT_ID = os.environ.get('SAXO_CLIENT_ID')
CLIENT_SECRET = os.environ.get('SAXO_CLIENT_SECRET')
REDIRECT_URI = 'http://localhost:5001/callback'
AUTH_URL = 'https://live.logonvalidation.net/authorize'
TOKEN_URL = 'https://live.logonvalidation.net/token'

STATE = ''.join(random.choices(string.ascii_letters + string.digits, k=24))

@app.route('/')
def login():
    if not hasattr(login, 'has_opened'):
        login.has_opened = True
        auth_request_url = (f"{AUTH_URL}?response_type=code&client_id={CLIENT_ID}"
                            f"&redirect_uri={REDIRECT_URI}&state={STATE}")
        return redirect(auth_request_url)
    return "Login page has already been opened.", 400
@app.route('/callback')
def callback():
    code = request.args.get('code')
    state = request.args.get('state')

    if state != STATE:
        return "State mismatch error", 400

    token_response = requests.post(TOKEN_URL, headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    }, data={
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    })

    token_json = token_response.json()
    access_token = token_json.get('access_token')
    if access_token:
        with open('data/saxo_api_key.txt', 'w') as f:
            f.write(access_token)
        
        shutdown_server()
        return "Access token saved. You can close this window.", 200
    else:
        return "Failed to retrieve access token", 400

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        print('Server shutdown function not found. Forcing shutdown...')
        os._exit(0)
    func()

def open_browser():
    webbrowser.open(f"http://localhost:5001/")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True, port=5001)