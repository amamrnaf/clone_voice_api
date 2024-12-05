from flask import Flask, request, jsonify
from flask_cors import CORS
from TTS.api import TTS
import boto3
import subprocess
import os
from functools import wraps
from botocore.exceptions import NoCredentialsError
import logging

# Flask app setup
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Logs to a file named "app.log"
        logging.StreamHandler()          # Logs to the console
    ]
)
logger = logging.getLogger("TTS_API")

# API Key
API_KEY = "G7x9mVt2Q5bK8Jp4S1Zc"

# Decorator to enforce API key
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key and key == API_KEY:
            logger.info(f"Valid API key used for {request.path}")
            return f(*args, **kwargs)
        else:
            logger.warning(f"Invalid or missing API key for {request.path}")
            return jsonify({"message": "Invalid or missing API key"}), 403
    return decorated_function

# Initialize TTS


# AWS S3 Configuration
S3_BUCKET = "my-tts-audio-files"
S3_REGION = "us-east-1"
s3_client = boto3.client("s3", region_name=S3_REGION)

# Helper function: Upload to S3
def upload_to_s3(file_name, key):
    try:
        s3_client.upload_file(file_name, S3_BUCKET, key)
        file_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
        logger.info(f"File uploaded to S3: {file_url}")
        return {"success": True, "url": file_url}
    except NoCredentialsError:
        logger.error("S3 credentials not available")
        return {"success": False, "error": "Credentials not available"}
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return {"success": False, "error": str(e)}

# Endpoint: Upload a new speaker
@app.route("/upload_speaker", methods=["POST"])
@require_api_key
def upload_speaker():
    if "file" not in request.files or "speaker_name" not in request.form:
        logger.warning("Missing file or speaker_name in /upload_speaker")
        return jsonify({"success": False, "message": "File and speaker_name are required!"}), 400

    file = request.files["file"]
    speaker_name = request.form["speaker_name"]
    input_file_path = f"/tmp/{file.filename}"
    speaker_file_path = f"/home/ubuntu/speakers/{speaker_name}.wav"

    # Save the uploaded file temporarily
    file.save(input_file_path)

    # Process the file to match TTS requirements
    try:
        process_audio(input_file_path, speaker_file_path)
        logger.info(f"Speaker {speaker_name} uploaded successfully")
        return jsonify({"success": True, "message": f"Speaker {speaker_name} uploaded successfully!"})
    except Exception as e:
        logger.error(f"Error uploading speaker: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        os.remove(input_file_path)  # Cleanup temporary file

# Helper function: Process uploaded speaker file
def process_audio(input_file, speaker_file_path):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-ar", "44100",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        speaker_file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise Exception(result.stderr)

# Endpoint: Generate TTS
@app.route("/generate_tts", methods=["POST"])
@require_api_key
def generate_tts():
    data = request.json
    text = data.get("text")
    speaker_name = data.get("speaker_name")

    if not text or not speaker_name:
        logger.warning("Missing text or speaker_name in /generate_tts")
        return jsonify({"success": False, "message": "Text and speaker_name are required!"}), 400

    speaker_file = f"/home/ubuntu/speakers/{speaker_name}.wav"
    if not os.path.exists(speaker_file):
        logger.error(f"Speaker {speaker_name} not found")
        return jsonify({"success": False, "message": f"Speaker {speaker_name} not found!"}), 404

    output_file = "output.wav"
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to("cuda")
        # Generate TTS
        logger.info(f"Generating TTS for text: '{text}' using speaker: '{speaker_name}'")
        tts.tts_to_file(text=text, file_path=output_file, speaker_wav=speaker_file, language="en")

        # Upload TTS output to S3
        s3_key = f"tts_outputs/{speaker_name}.wav"
        s3_response = upload_to_s3(output_file, s3_key)
        file_url=s3_response["url"]
        if not s3_response["success"]:
            raise Exception(s3_response["error"])
        
        return jsonify({"success": True, "url": file_url})
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)  

if __name__ == "__main__":
    
    os.makedirs("/home/ubuntu/speakers", exist_ok=True)
    app.run(debug=True, threaded=True)
