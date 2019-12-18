from PIL import Image
import numpy as np
import base64
import tensorflow as tf
import os

    if request.method =='POST':
        #画像データの取得
        form = DocumentForm(request.POST, request.FILES)
        
        if form.is_valid():
            form.save()

            return redirect('index')

    else:
        return redirect('index.html')



     #簡易エラーチェック（拡張子）
    root, ext = os.path.splitext(pic.name)
    if ext != '.png':
        message ="【ERROR】png以外の拡張子ファイルが指定されています。"
        return render(request, 'index.html', {
                        "message": message,
                        })


    #(入力画像,キャラ名)

    img = Image.open(pic)
    img = img.resize((150, 150))
    data = np.asarray(img)
    data = np.expand_dims(data, axis=0)
    label = anime(data) #キャラ名
        
    #入力画像とキャラ名をまとめて格納→入力画像でなく、推測したキャラの画像を表示したい。

    if anime.features[0,0]== 1:
        src = Image.open('./aniface/character_img/0')
        return src     
    elif anime.features[0,1]== 1:
        src = Image.open('./aniface/character_img/1')
        return src
    else:
        src = Image.open('./aniface/character_img/2')
        return src
    src = base64.b64encode(src)
    src = str(src)[2:-1]
    result = [src, label]

    context = {
        'result': result,
        }

    return render(request, 'result.html', context)


def anime(data):
    categories = []

    global graph
    #予測
    with graph.as_default():

        features = model.predict(data)
        #features[0,i] if features[0,i]==1: return categories[i]
        #予測結果によって処理を分ける
        if features[0,0] == 1:
            return "shana"

        elif features[0,1] == 1:
            return "haruhi"

        else:
            return "hatsune"



class Document(models.Model):
    photo = models.ImageField(upload_to='')








from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    tf.keras.backend.set_session(session)

#学習モデルのロード
    current_dir = os.getcwd()
    model_path = current_dir + '/model/class3.hdf5'
    model = load_model(model_path)
    graph = tf.compat.v1.get_default_graph()
