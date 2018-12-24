import tensorflow as tf

from en2kr import En2kr

tf.app.flags.DEFINE_boolean("train", False, "학습모드. 테스트를 실행하지 않습니다.")
FLAG = tf.app.flags.FLAGS


def main(_):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
