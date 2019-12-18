from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

graph = tf.get_default_graph()
 

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 150 # 画像サイズ
    MODEL_FILE_PATH = './model/class3.hdf5' # モデルファイル
    classes = ["サトシ","のび太","まる子ちゃん","しんのすけ","名探偵コナン","浅倉南","綾波レイ","夜神月","孫悟空","磯野カツオ","毛利蘭","磯野サザエ","しずかちゃん","坂田銀時","神楽","ルフィ","ナミ（ワンピース）"]
    num_classes = len(classes)

    # 引数から画像ファイルを参照して読み込む
    def predict(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)

            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image) / 255.0
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            # print(self.classes[predicted], percentage)
            return self.classes[predicted], percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' + img.file.content_type + ';base64,' + base64_img