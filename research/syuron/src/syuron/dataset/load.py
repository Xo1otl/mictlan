import numpy as np
import tensorflow_datasets as tfds
from workspace import path
from .dataset import Dataset


def preprocess_numpy(image, label):
    """NumPyを使った前処理関数"""
    # 画像をfloat32にキャストして正規化
    image = image.astype(np.float32) / 255.0
    # 画像を1次元に平坦化 (28x28=784)
    image = image.reshape(-1)  # または image.flatten()
    # ラベルをone-hot変換 (10クラス)
    label_onehot = np.zeros(10, dtype=np.float32)
    label_onehot[label] = 1.0
    return image, label_onehot


def prepare_mnist_mlp_dataset_numpy():
    """NumPyベースでMNISTデータセットを準備"""
    # TensorFlow Datasetsでデータを読み込み、NumPy配列に変換
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds_numpy = tfds.as_numpy(ds)

    # NumPy配列のリストとして前処理済みデータを格納
    processed_images = []
    processed_labels = []

    for image, label in ds_numpy:
        proc_image, proc_label = preprocess_numpy(image, label)
        processed_images.append(proc_image)
        processed_labels.append(proc_label)

    # NumPy配列に変換
    images_array = np.array(processed_images)
    labels_array = np.array(processed_labels)

    return images_array, labels_array


# バッチ処理版（メモリ効率を考慮）
def prepare_mnist_mlp_dataset_numpy_batched(batch_size=1000):
    """バッチ処理でメモリ効率を向上させた版"""
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds_numpy = tfds.as_numpy(ds)

    images_batches = []
    labels_batches = []
    current_images = []
    current_labels = []

    for image, label in ds_numpy:
        proc_image, proc_label = preprocess_numpy(image, label)
        current_images.append(proc_image)
        current_labels.append(proc_label)

        if len(current_images) >= batch_size:
            images_batches.append(np.array(current_images))
            labels_batches.append(np.array(current_labels))
            current_images = []
            current_labels = []

    # 残りのデータを処理
    if current_images:
        images_batches.append(np.array(current_images))
        labels_batches.append(np.array(current_labels))

    # 全バッチを結合
    all_images = np.concatenate(images_batches, axis=0)
    all_labels = np.concatenate(labels_batches, axis=0)

    return all_images, all_labels


def load(filepath: path.Path, batch_size=1024) -> Dataset:
    save_path = filepath.abs()
    dataset = tfds.load(save_path, batch_size=batch_size)
    ds_np = tfds.as_numpy(dataset)
    return ds_np

# if __name__ == '__main__':
#     dataset = prepare_mnist_mlp_dataset()

#     save_path = path.Path(
#         'research/syuron/dataset/mnist_flattened.tfrecord').abs()

#     dataset.save(save_path)
#     print("Dataset saved to:", save_path)

#     loaded_dataset = tf.data.Dataset.load(save_path)
#     for element in loaded_dataset.take(1):
#         print("Loaded element:", element)
