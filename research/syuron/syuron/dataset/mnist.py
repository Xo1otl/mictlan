import tensorflow as tf
import tensorflow_datasets as tfds
from workspace import path
from .dataset import Dataset


def prepare_mnist_mlp_dataset():
    # as_supervised=Trueにより、各サンプルは (image, label) のタプルとして取得される
    ds = tfds.load('mnist', split='train', as_supervised=True)

    def preprocess(image, label):
        # 画像をfloat32にキャストして正規化
        image = tf.cast(image, tf.float32) / 255.0  # type: ignore
        # 画像を1次元に平坦化 (28x28=784)
        image = tf.reshape(image, [-1])
        # ラベルをone-hot変換 (10クラス)
        label = tf.one_hot(label, depth=10)
        return image, label

    ds = ds.map(preprocess)  # type: ignore
    return ds


def load_mnist(batch_size=1024) -> Dataset:
    save_path = path.Path(
        'research/syuron/dataset/mnist_flattened.tfrecord').abs()
    dataset = tf.data.Dataset.load(save_path)
    ds = dataset.batch(batch_size)
    ds_np = tfds.as_numpy(ds)
    return ds_np


if __name__ == '__main__':
    dataset = prepare_mnist_mlp_dataset()

    save_path = path.Path(
        'research/syuron/dataset/mnist_flattened.tfrecord').abs()

    dataset.save(save_path)
    print("Dataset saved to:", save_path)

    loaded_dataset = tf.data.Dataset.load(save_path)
    for element in loaded_dataset.take(1):
        print("Loaded element:", element)
