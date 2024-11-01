#!/usr/bin/env python
# coding: utf-8


from flask import Flask, request, jsonify, send_file, Response
from requests_toolbelt.multipart import MultipartEncoder
import numpy as np
from scipy.spatial.distance import cosine
import os
from PIL import Image
from io import BytesIO
import pickle
import requests
import json
from U_Model import Unet
import cv2
import io
import base64
from flask_cors import CORS

#Tensorflow 관련
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') #GPU환경이 아닌 CPU환경에서 진행하기 위함
from tensorflow.keras import backend as K
import torch

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": ["http://localhost:3000",'https://carumuch-frontend.vercel.app']}}, supports_credentials=True)
# app 실행 시 VGG16 모델 로드 (지연 로드 방식 참고해볼것(추후))
base_model = VGG16(weights='imagenet')
v_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

#Unet model load
device = torch.device('cpu')
labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
models = []
n_classes = 2
for label in labels:
    model_path = f'models/[DAMAGE][{label}]Unet.pt'

    model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    models.append(model)


#헬스체크
@app.route('/health-check', methods=['GET'])
def healthCheck():
    return 'OK'


@app.route('/segment',methods=['POST'])
def segment():
    data = request.json
    image_url = data.get('url')
    def process_image(img_url):
        img = requests.get(img_url)
        img_array = np.asarray(bytearray(img.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_input = img / 255.
        img_input = img_input.transpose([2, 0, 1])
        img_input = torch.tensor(img_input).float().to(device)
        img_input = img_input.unsqueeze(0)
        return img,img_input
    def overlay(img,img_input):
        overlay = np.zeros_like(img)
        colors = [
            (255, 0, 0),    # Breakage_3 -> 빨간색
            (0, 255, 0),    # Crushed_2 -> 초록색
            (0, 0, 255),    # Scratch_0 -> 파란색
            (255, 255, 0)   # Seperated_1 -> 노란색
        ]  # 각 모델의 결과를 표현할 색상

        for idx, model in enumerate(models):
            with torch.no_grad():
                output = model(img_input)  # 모델로 예측
                prediction = torch.argmax(output, dim=1).cpu().numpy()  # 클래스 예측 결과 (0 또는 1)

            # 예측 결과를 시각화하기 위해 색상 적용
            mask = prediction[0]  # 배치 차원 제거
            for c in range(3):
                overlay[:, :, c] = np.where(mask == 1, colors[idx][c], overlay[:, :, c])

        # 원본 이미지와 오버레이된 결과를 합성
        alpha = 0.3  # 투명도 설정
        output_image = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        return output_image
    #이미지 처리 및 오버레이
    img, img_input = process_image(image_url)
    output_image = overlay(img, img_input)

    # 이미지를 바이트 스트림으로 변환하여 반환
    output_pil_image = Image.fromarray(output_image)
    img_io = io.BytesIO()
    output_pil_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

@app.route('/classify', methods=['POST'])
def classify():
    #전처리 및 VGG16기반 이미지 특징값 추출
    def process_image(image_url):
        # 이미지 처리 함수
        img = requests.get(image_url)
        img = Image.open(BytesIO(img.content))
        img = img.resize((224, 224))
        try:
            # img가 이미 로드된 이미지일 경우 바로 처리
            img = img.resize((224, 224))  # 이미지를 VGG16 입력 크기로 조정
            img_data = image.img_to_array(img)  # 이미지를 배열로 변환
            img_data = np.expand_dims(img_data, axis=0)  # 배치 차원 추가
            img_data = preprocess_input(img_data)  # VGG16 모델에 맞게 전처리
            
            # 모델을 사용하여 특징 추출
            features = v_model.predict(img_data)
            return features.flatten()  # 특징을 1차원 배열로 반환
        except Exception as e:
            print(f"Error loading image: {e}")

    #Cosine 유사도 기반 이미지 비교
    def npath_compare_images(img_features, features_dict):
        new_features = img_features
        most_similar_image = None
        highest_similarity = -1
        
        for filename, features in features_dict.items():
            similarity = 1 - cosine(new_features, features)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_image = filename
                
        return most_similar_image, highest_similarity

    #가장 가까운 cosine값에 대한 견적서 제출
    def close_cost(cost_file_path, most_similar_image,manufacturer):
        total_cost = 0
        list_repair = []
        
        json_file_name = most_similar_image.replace('.jpg', '.json')
        json_file_path = os.path.join(cost_file_path, json_file_name)
        print(json_file_path)
        if '\\' in json_file_path:
            json_file_path = json_file_path.replace("\\", "/")
        print(json_file_path)
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if '수리내역' in data:
            repair_details = data['수리내역']
            for item in repair_details:
                if '손해사정후' in item:
                    # 기아 차량의 경우 (손해사정후 데이터 있음)
                    part_cost_str = item['손해사정후'].get('부품가격', '0')
                    if part_cost_str is None or part_cost_str == '':
                        part_cost_str = '0'
                    part_cost = int(part_cost_str.replace(',', ''))

                    labor_cost_str = item['손해사정후'].get('공임', '0')
                    if labor_cost_str is None or labor_cost_str == '':
                        labor_cost_str = '0'
                    labor_cost = int(labor_cost_str.replace(',', ''))
                else:
                    # 현대 차량의 경우 (손해사정후 데이터 없음)
                    part_cost_str = item.get('부품가격', '0')
                    if part_cost_str is None or part_cost_str == '':
                        part_cost_str = '0'
                    part_cost = int(part_cost_str.replace(',', ''))

                    labor_cost_str = item.get('공임', '0')
                    if labor_cost_str is None or labor_cost_str == '':
                        labor_cost_str = '0'
                    labor_cost = int(labor_cost_str.replace(',', ''))

                total_cost += part_cost + labor_cost

                # 작업 항목 및 부품명 리스트에 추가
                list_repair.append(item['작업항목 및 부품명'])
                
            #print(list_repair, total_cost)
        else:
            print("'수리내역' 키가 JSON 파일에 없습니다.")
        
        return list_repair, total_cost
        
        
        ## 세그멘테이션용 이미지 전처리
    def seg_process_image(img_url):
            img = requests.get(img_url)
            img_array = np.asarray(bytearray(img.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img_input = img / 255.
            img_input = img_input.transpose([2, 0, 1])
            img_input = torch.tensor(img_input).float().to(device)
            img_input = img_input.unsqueeze(0)
            return img,img_input
        # 파손부위 판별 이미지
    def overlay(img,img_input):
            overlay = np.zeros_like(img)
            colors = [
                (255, 0, 0),    # Breakage_3 -> 빨간색
                (0, 255, 0),    # Crushed_2 -> 초록색
                (0, 0, 255),    # Scratch_0 -> 파란색
                (255, 255, 0)   # Seperated_1 -> 노란색
            ]  # 각 모델의 결과를 표현할 색상

            for idx, model in enumerate(models):
                with torch.no_grad():
                    output = model(img_input)  # 모델로 예측
                    prediction = torch.argmax(output, dim=1).cpu().numpy()  # 클래스 예측 결과 (0 또는 1)

                # 예측 결과를 시각화하기 위해 색상 적용
                mask = prediction[0]  # 배치 차원 제거
                for c in range(3):
                    overlay[:, :, c] = np.where(mask == 1, colors[idx][c], overlay[:, :, c])

            # 원본 이미지와 오버레이된 결과를 합성
            alpha = 0.3  # 투명도 설정
            output_image = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            return output_image
    
    data = request.json
    image_url = data.get('url')
    manufacturer = data.get('manufacturer')
    
    if manufacturer == 'kia':
        img_features = process_image(image_url)
        pickle_filename = './kia_features_dict.pkl'
        with open(pickle_filename, 'rb') as f:
            loaded_df = pickle.load(f)
        most_similar_image, highest_similarity = npath_compare_images(img_features, loaded_df)
        k_cost_file_path = './kia_견적서'
        list_repair, total_cost = close_cost(k_cost_file_path, most_similar_image, manufacturer)
        del(loaded_df)

        # 세그멘테이션 처리
        img, img_input = seg_process_image(image_url)
        output_image = overlay(img, img_input)

        # 이미지를 바이트 스트림으로 변환하여 Base64로 인코딩
        try:
            output_pil_image = Image.fromarray(output_image)
            img_io = io.BytesIO()
            output_pil_image.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # JSON 응답 데이터 구성
            response_data = {
                "list_repair": list_repair,
                "total_cost": total_cost,
                "segmented_image": img_base64
            }

            # 디버깅 메시지 추가
            print("img_base64 생성 성공:", img_base64[:100])  # 인코딩된 이미지의 일부 출력

            return jsonify(response_data)

        except Exception as e:
            # 오류 발생 시 로그에 출력하고 오류 메시지 반환
            print(f"이미지 인코딩 오류: {e}")
            return jsonify({"error": "이미지 인코딩에 실패했습니다."}), 500
        
    elif manufacturer == 'hyundai':
    # 이미지 처리 함수 적용 (특징 추출)
        img_features = process_image(image_url)
        pickle_filename = './h_features_dict.pkl'
        
        # 견적서 데이터 로드 및 유사도 비교
        with open(pickle_filename, 'rb') as f:
            loaded_df = pickle.load(f)
        most_similar_image, highest_similarity = npath_compare_images(img_features, loaded_df)
        print(most_similar_image)
        h_cost_file_path = './hyundai_견적서'
        list_repair, total_cost = close_cost(h_cost_file_path, most_similar_image, manufacturer)
        del(loaded_df)

        # 세그멘테이션 부분 - 이미지를 처리 및 오버레이 적용
        img, img_input = seg_process_image(image_url)
        output_image = overlay(img, img_input)

        # 이미지를 바이트 스트림으로 변환하여 Base64로 인코딩
        try:
            output_pil_image = Image.fromarray(output_image)
            img_io = io.BytesIO()
            output_pil_image.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # JSON 응답 데이터 구성
            response_data = {
                "list_repair": list_repair,
                "total_cost": total_cost,
                "segmented_image": img_base64
            }

            return jsonify(response_data)

        except Exception as e:
            # 오류 발생 시 로그에 출력하고 오류 메시지 반환
            print(f"이미지 인코딩 오류: {e}")
            return jsonify({"error": "이미지 인코딩에 실패했습니다."}), 500

@app.after_request
def after_request(response):
    origin = request.headers.get("Origin")
    allowed_origins = ["http://localhost:3000", "https://carumuch-frontend.vercel.app"]
    if origin in allowed_origins:
        response.headers.add("Access-Control-Allow-Origin", origin)
    #response.headers.add('Access-Control-Allow-Origin', ['http://localhost:3000','https://carumuch-frontend.vercel.app'])
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response
if __name__ == '__main__':
    # 서버를 5000 포트에서 실행
    app.run(port=5000,debug=True)






