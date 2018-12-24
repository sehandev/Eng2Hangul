import tensorflow as tf
import numpy as np

from hangul_decompose import decompose

class En2kr:
    def __init__(self, session):
        self.sess = session
        self.learning_rate = 1e-2
        self.n_input = 26 + 3  # 29 = Alphabet + S/E/P
        self.n_hidden = 128
        self.n_class = 19 + 21 + 1 + 3  # 44 = 초성 + 중성 + 종성없을때 + S/E/P
        self.total_epoch = 100
        
        self.build_network()
        self.train_model()
        
    def load_data(self):
        with open('en2kr_correct.txt', 'r') as f:
            tmp = f.read().splitlines()
        
        seq_data = list()
        for line in tmp:
            data = list()
            front = line.split(' ')[0]
            back  = line.split(' ')[1]
            
            for letter in back:
                a, b, c = decompose(letter)
                back_decomposed = a + b + c
            data.append(front)
            data.append(back_decomposed)
            seq_data.append(data)
        
        return seq_data
        
    def make_batch(self, seq_data):
        en_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
        num_en = {n: i for i, n in enumerate(en_arr)}
        
        CHO = [u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ', u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ']
        JOONG = [u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ', u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ']
        JAMO = ['S', 'E', 'P'] + CHO + JOONG
        kr_arr = [c for c in JAMO]
        num_kr = {n: i for i, n in enumerate(kr_arr)}
        
        input_batch = list()
        output_batch = list()
        target_batch = list()

        for seq in seq_data:
            # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
            input = [num_en[n] for n in seq[0]]
            # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
            output = [num_kr[n] for n in ('S' + seq[1])]
            # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
            target = [num_kr[n] for n in (seq[1] + 'E')]

            input_batch.append(np.eye(self.n_input)[input])
            output_batch.append(np.eye(self.n_class)[output])
            target_batch.append(np.eye(self.n_class)[target])

        return input_batch, output_batch, target_batch
    
    def build_network(self):
        self.enc_input = tf.placeholder(tf.float32, [None, None, self.n_input])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.n_input])
        # [batch size, time steps, input size]
        self.targets = tf.placeholder(tf.int64, [None, None])  # [batch size, time steps]

        outputs = self.build_cells(0.5)
        model = tf.layers.dense(outputs, self.n_class, activation=None)
        self.build_operators(model)

    def build_cells(self, keep_prob):
        # 인코더 셀을 구성한다.
        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, name='basic_lstm_cell')
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=keep_prob)
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input,
                                                    dtype=tf.float32)
        # 디코더 셀을 구성한다.
        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, name='basic_lstm_cell')
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input,
                                                    initial_state=enc_states,
                                                    dtype=tf.float32)
        return outputs
    
    def build_operators(self, model):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.targets))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    # 신경망 모델 학습
    def train_model(self):
        self.sess.run(tf.global_variables_initializer())
        
        seq_data = self.load_data()

        input_batch, output_batch, target_batch = self.make_batch(seq_data)
        print(target_batch)

        for epoch in range(self.total_epoch):
            _, loss = self.sess.run([self.optimizer, self.cost],
                               feed_dict={self.enc_input: input_batch,
                                          self.dec_input: output_batch,
                                          self.targets: target_batch})

            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        print('최적화 완료!')
        
        
t = En2kr(tf.Session())
