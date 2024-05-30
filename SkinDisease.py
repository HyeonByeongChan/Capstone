import numpy as np
TF_ENABLE_ONEDNN_OPTS = 0
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import sys
import pandas as pd

def load_image(img_path, show=False):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    if show:
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.show()
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def ensemble_predict(models, img):
    # 여러 모델의 예측을 평균 내어 최종 예측을 수행합니다.
    predictions = np.array([model.predict(img) for model in models])
    avg_predictions = np.mean(predictions, axis=0)
    class_labels = ['flea_allergy', 'hotspot', 'mange', 'ringworm']
    predicted_class = class_labels[np.argmax(avg_predictions)]
    return predicted_class, avg_predictions

# 이미지 파일 경로 직접 지정
# img_path = 'mange.jpg'  # 사용자의 이미지 경로로 변경

# 이미지 로드 및 출력


# 모델들을 로드 (적절한 모델 파일 경로 지정 필요)
models = [
    tf.keras.models.load_model('C:/Users/skm99/OneDrive/Desktop/SkinDisease-S.h5'),
    tf.keras.models.load_model('C:/Users/skm99/OneDrive/Desktop/SkinDisease2-S.h5'),
    # tf.keras.models.load_model('SkinDisease3.h5'),
    # tf.keras.models.load_model('SkinDisease4.h5'),
    tf.keras.models.load_model('C:/Users/skm99/OneDrive/Desktop/SkinDisease5-S.h5')
]

# 이미지에 대해 앙상블 예측 수행
# label, predictions = ensemble_predict(models, img)

# print("\nThe image is detected as " + label)
# print('\n')
# print('\n')


if __name__ == "__main__":
    image_path = "C:/Users/skm99/OneDrive/Desktop/mange.jpg"
    # sys.argv[1]  # 이미지 경로는 명령줄 인수로 받음
    img = load_image(image_path)
    label, predictions = ensemble_predict(models, img)
    print(label)