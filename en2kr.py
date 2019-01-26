import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from hangul_decompose import decompose

# __init__
learning_rate = 1e-3
n_hidden = 512
total_epoch = 10000
t_epoch = 20
word_limit = 5

char_arr = [c for c in 'EPabcdefghijklmnopqrstuvwxyzㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅚㅙㅛㅜㅝㅞㅟㅠㅡㅣ']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)


def pad(array, max_length):
    while len(array) < max_length:
        array.append(1)
    return array


def make_batch(seq_data, limit):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        if len(seq[0]) <= limit and len(seq[1]) <= limit*2:
            # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
            input = [num_dic[n] for n in seq[0]]
            input = pad(input, limit+1)
            # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
            output = [num_dic[n] for n in ('E' + seq[1])]
            output = pad(output, limit*2+1)
            # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
            target = [num_dic[n] for n in (seq[1] + 'E')]
            target = pad(target, limit*2+1)

            input_batch.append(np.eye(dic_len)[input])
            output_batch.append(np.eye(dic_len)[output])
            target_batch.append(np.eye(dic_len)[target])
    return input_batch, output_batch, target_batch


# build_network
global_step = tf.Variable(0, trainable=False, name='global_step')
enc_input = tf.placeholder(tf.float32, [None, None, dic_len], name='enc_input')  # [batch size, time steps, input size]
dec_input = tf.placeholder(tf.float32, [None, None, dic_len], name='dec_input')  # [batch size, time steps, class size]
targets = tf.placeholder(tf.float32, [None, None, dic_len], name='targets')  # [batch size, time steps, class size]
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout : 0.0 ~ 1.0

# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, name='basic_lstm_cell')
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=keep_prob)
    _, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)
# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, name='basic_lstm_cell')
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)
model = tf.layers.dense(outputs, dic_len, activation=tf.nn.relu)
for i in range(20):
    model = tf.layers.dense(model, dic_len, activation=tf.nn.relu)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=targets, name='train_softmax'))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, global_step=global_step)

# tensor_board
tf.summary.scalar('cost', cost)

# train_model
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

with open('en2kr.txt', 'r') as f:
    tmp = f.read().splitlines()

seq_data = []
for line in tmp:
    data = []
    front = line.split(' ')[0]
    back  = line.split(' ')[1]

    back_decomposed = str()
    for letter in back:
        a, b, c = decompose(letter)
        back_decomposed += a + b + c
    data.append(front)
    data.append(back_decomposed)
    seq_data.append(data)

input_batch, output_batch, target_batch = make_batch(seq_data, word_limit)

train_input, test_input, train_output, test_output, train_target, test_target = train_test_split(input_batch, output_batch, target_batch, test_size=0.2, random_state=0)

# merge tensorboard
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)


def predict(word):
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: test_input,
                                 dec_input: test_output,
                                 targets: test_target,
                                 keep_prob: 1.0})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    print(result[0])
    decoded = [char_arr[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    try:
        end = decoded.index('E')
        predicted = ''.join(decoded[:end])
    except:
        try:
            end = decoded.index('P')
            predicted = ''.join(decoded[:end])
        except:
            end = 5
            predicted = ''.join(decoded[:end])

    return predicted


start_step = sess.run(global_step)


for epoch in range(start_step, total_epoch+1):
    sess.run(optimizer,
            feed_dict={enc_input: train_input,
                dec_input: train_output,
                targets: train_target,
                keep_prob: 0.5})

    if epoch % t_epoch == 0:
        loss = sess.run(cost, feed_dict={enc_input: train_input,
            dec_input: train_output,
            targets: train_target,
            keep_prob: 0.5})
        print('Epoch:', '%05d' % epoch, 'Train cost =', '{:.6f}'.format(loss))
        loss = sess.run(cost, feed_dict={enc_input: test_input,
                    dec_input: test_output,
                    targets: test_target,
                    keep_prob: 1.0})
        print('Test cost =', '{:.6f}'.format(loss))
        saver.save(sess, './model/dnn.ckpt', global_step=global_step)

        summary = sess.run(merged, feed_dict={enc_input: test_input,
                                      dec_input: test_output,
                                      targets: test_target,
                                      keep_prob: 1.0})
        writer.add_summary(summary, global_step=sess.run(global_step))

        print('=== 테스트 ===')

        print('aala ->', predict('aala'))
        print('word ->', predict('word'))
        print('world ->', predict('world'))
        print('sehan ->', predict('sehan'))
        print('\n\n')


print('최적화 완료!')
