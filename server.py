from flask import Flask, request, abort
import os
from inference import VideoGen
from minio import Minio
from datetime import timedelta
import requests
from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv


class S3:
    BUCKET = "hackathon"

    def __init__(self):
        load_dotenv()
        access_key = os.getenv("MINIO_URL")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        self.s3client = Minio(
            "s3.jiaxiaodong.com",
            access_key=access_key,
            secret_key=secret_key,
        )
        if not self.s3client.bucket_exists(self.BUCKET):
            raise Exception("no bucket")

    def upload(self, name, path):
        self.s3client.fput_object(self.BUCKET, name, path)
        print(f"uploaded {path}")

    def get_link(self, name):
        url = self.s3client.presigned_get_object(self.BUCKET, name, expires=timedelta(days=1))
        return url

s3 = S3()
videogen = VideoGen()
app = Flask(__name__)
pool = ThreadPool(processes=1)

@app.route("/api/v1/hello_world")
def hello_world():
    return "Hello world", 200

@app.route("/api/v1/test_callback", methods=["POST"])
def test_callback():
    req = request.json
    query = req.get("query", "")
    callback_url = req.get("callback_url", "")
    data = { "url": "test test 1 2 3"}
    requests.post(callback_url, json=data)
    return query, 200

def generate_video(query, callback_url):
    print("generating video")
    video_path, filename = videogen.llm_to_video(query)
    s3.upload(filename, video_path)
    url = s3.get_link(filename)
    print(url)
    requests.post(callback_url, json={'url': url})

@app.route("/api/v1/prompt", methods=["POST"])
def prompt():
    req = request.json
    if "query" not in req:
        return "query cannot be empty", 401
    query = req["query"]
    callback_url = req.get("callback_url", "")
    pool.apply(generate_video, args=[query, callback_url])
    return url, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
