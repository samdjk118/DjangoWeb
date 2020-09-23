from django.shortcuts import render

from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from tensorflow.python.keras.backend import set_session
# from keras.models import load_model
from tensorflow.python.keras.models import load_model

import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions

import io
import numpy as np
import cv2

from skimage import io as skio
from skimage import transform


sess = tf.Session()
set_session(sess)

graph = tf.get_default_graph()
model = load_model('Sex_Age/model_inceptionv3_celeb_gender.h5')
print("load gender_model...")

# Create your views here.
def home(  request  ):
    return render(request, 'Sex_Age/home.html')

import requests
import re

# Create your views here.
# @csrf_exempt

@csrf_exempt
def api_classify_image_url(request):
    url = request.POST.get('input_img_url')
    ximg = prepare_image(url)
    response = classify_image(ximg)
    print(response)
    return JsonResponse(response)

# 依據url path讀取圖檔 也可以讀取 inMemory 的圖檔
def prepare_image(img_url):
    im = skio.imread(img_url)
    im = cv2.resize(im, (178, 218)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)
    return im


def classify_image(img):
    with graph.as_default():
        set_session(sess)
        y = model.predict(img)
    pred = {}  # 輸出dictionary
    Sex_name = ["Female","Male"]
    pred["predictions"] = []

    for i in range(2):
        result = {"label": Sex_name[i], "proba": float(y[0][i])}
        pred["predictions"].append(result)

    return pred


@csrf_exempt
def api_classify_image(  request  ):
    if request.method == 'POST':

        img = request.FILES["upload_image"] # InMemoryUploadedFile

        ximg = prepare_image( img ) # 不能讀取送來的2進位檔案

        result = classify_image(ximg) #預測 得到結果
        return JsonResponse(result)
    return JsonResponse({"predictions":"None"})

@csrf_exempt
def api_classify_image_upload(request):

    # 上傳過來的檔案存放於記憶體中
    # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
    img_inMemory = request.FILES.get('upload_image')
    # print(type(img_inMemory))

    # 圖檔前處理 預測其類別
    ximg = prepare_image(img_inMemory)
    response = classify_image(ximg)

    # Django存檔至media目錄
    # 檔案存放路徑c:/xx/xx/xx/media/target_image.jpg
    # 原有檔案不會被覆蓋  新檔名會使用原檔名加上隨機碼 多使用者沒問題
    fs = FileSystemStorage()
    file_info = fs.save('target_image.jpg', img_inMemory)

    # 讓前端可以取得影像檔案顯示
    # uploaded_file_url 是: /media/target_image_FhObuDy.jpg
    uploaded_file_url = fs.url(file_info)
    print(uploaded_file_url)
    response['img_url'] = uploaded_file_url

    # 若要取得完整的伺服器檔案路徑c:/xx/xx/xx/media/target_image.jpg
    # image_path = fs.path(file_info)

    # print(response)
    return JsonResponse(response)
