from flask import Flask, request
import requests
import threading
import time

app = Flask(__name__)

@app.route("/", methods=["POST"])
def get_callback():
    req = request.json
    print(req["url"]) # s3 url
    return "OK", 200

def call_server():
    body = {
        "query": "Pretend you are President Kennedy. What are your opinions on the North Koreans?",
        "callback_url": "http://127.0.0.1:5000"
    }
    # change to test.jiaxiaodong.com when deploying
    requests.post("http://localhost:7860/api/v1/prompt", json=body, allow_redirects=True)

def run():
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    # start client in the background before calling server, 
    # if not server might not be able to connect
    x = threading.Thread(target=run)
    x.start()
    time.sleep(1)
    call_server()
    x.join()
