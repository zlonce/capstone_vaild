import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
from person_detector import PersonDetector
from nsfw_detector import NSFWDetector
from text_detector import TextDetector
from get_tag import tagging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

STORE_SERVER_URL = os.getenv('STORE_SERVER_URL', 'http://store-service:8084/photo-store/photos')

def validate(image_path):
    logger.info(f"이미지 검증 시작: {image_path}")
    
    nsfw_detector = NSFWDetector(threshold=0.5)
    person_detector = PersonDetector('./person_detection.pth')
    text_detector = TextDetector()

    if nsfw_detector.detect(image_path):
        logger.warning("NSFW 콘텐츠 감지됨")
        return "nsfw"
    elif text_detector.detect(image_path):
        logger.warning("텍스트 감지됨")
        return "text"
    elif person_detector.detect(image_path):
        logger.warning("사람 감지됨")
        return "person"
    
    logger.info("검증 통과")
    return False

@app.route('/validate', methods=['POST'])
def validate_image():
    logger.info("이미지 검증 요청 받음")
    
    if 'image' not in request.files:
        logger.error("이미지 파일이 요청에 없음")
        return jsonify({'error': '이미지가 없습니다'}), 400
        
    file = request.files['image']
    original_data = request.form.get('data', '{}')
    
    try:
        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(temp_path)
        logger.info(f"이미지 임시 저장됨: {temp_path}")
        
        error = validate(temp_path)
        if error:
            logger.warning(f"유효성 검사 실패: {error}")
            os.remove(temp_path)
            return jsonify({'error': error}), 400

        with open(temp_path, 'rb') as img_file:
            image_stream = io.BytesIO(img_file.read())
            
        tag = tagging(image_stream)
        logger.info(f"생성된 태그: {tag}")
        
        try:
            data_dict = json.loads(original_data)
        except json.JSONDecodeError:
            data_dict = {}
            
        data_dict['tag'] = tag

        with open(temp_path, 'rb') as img_file:
            files = {
                'file': (file.filename, img_file, 'image/jpeg')
            }
            form_data = {
                'request': json.dumps(data_dict)
            }
            
            response = requests.post(STORE_SERVER_URL, files=files, data=form_data)
            response.raise_for_status()
        
        os.remove(temp_path)
        logger.info("임시 파일 삭제됨")
        
        return jsonify({
            'success': True,
            'tags': tag,
            'response': response.json()
        }), 200
        
    except Exception as e:
        logger.error(f"처리 중 에러 발생: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500
    

@app.route('/tag', methods=['POST'])
def process_image():
    logger.info("이미지 태깅 요청 받음")
    
    if 'image' not in request.files:
        logger.error("이미지 파일이 요청에 없음")
        return jsonify({'error': '이미지가 없습니다'}), 400

    file = request.files['image']
    original_data = request.form.get('data', '{}')
    
    try:
        image_data = file.read()
        image_path = io.BytesIO(image_data)
        
        tag = tagging(image_path)
        logger.info(f"생성된 태그: {tag}")

        try:
            data_dict = json.loads(original_data)
        except json.JSONDecodeError:
            data_dict = {}
        
        data_dict['tag'] = tag

        files = {
            'file': (file.filename, image_path, 'image/jpeg')
        }

        form_data = {
            'request': json.dumps(data_dict)
        }
        
        response = requests.post(STORE_SERVER_URL, files=files, data=form_data)
        response.raise_for_status()
        
        return jsonify({
            'success': True,
            'tags': tag,
            'response': response.json()
        }), 200

    except Exception as e:
        logger.error(f"처리 중 에러 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=True)