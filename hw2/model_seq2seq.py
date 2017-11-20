def seq2seq_model(latent_dim=256, num_encoder_tokens=4096, num_decoder_tokens=3000):
    encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_input')
    encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm', implementation=2)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # state_h = Lambda(lambda x: K.print_tensor(x))(state_h)
    # state_c = Lambda(lambda x: K.print_tensor(x))(state_c)
    #encoder_outputs = Lambda(lambda x: K.print_tensor(x))(encoder_outputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_input')
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm', implementation=2)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

def inference_model(latent_dim=256, num_encoder_tokens=4096, num_decoder_tokens=3000):
    encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_input')
    encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_input')

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def train(model, batch_size, epochs):
    max_length = 3000
    padded_length = 44
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

    def data_generator(features, labels, batch_size):
        batch_encoder_input = np.zeros((batch_size, features.shape[1], features.shape[2]))
        batch_decoder_input = np.zeros((batch_size, padded_length, max_length))
        batch_decoder_output = np.zeros((batch_size, padded_length, max_length))
        x_train_size = len(features)
        sample_index = 0
        shifted_labels = list(labels)
        for i, label in enumerate(labels):
            labels[i] = pad_sequences(label, padding='post', maxlen=44)
            shifted_labels[i] = pad_sequences(list(map(lambda x: [2] + x, label)), padding='post', maxlen=44)
        while True:
            for i in range(batch_size):
                caption_index = np.random.randint(len(labels[sample_index]))
                batch_encoder_input[i] = features[sample_index]
                batch_decoder_input[i] = to_categorical(shifted_labels[sample_index][caption_index], 3000)
                batch_decoder_output[i] = to_categorical(labels[sample_index][caption_index], 3000)
                sample_index += 1
                if sample_index >= x_train_size:
                    sample_index = 0
            yield [batch_encoder_input, batch_decoder_input], batch_decoder_output

    import time
    start_time = time.time()
    model.fit_generator(
        data_generator(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=1450/batch_size,
        epochs=epochs,
        verbose=1)
    end_time = time.time()
    seconds = (end_time - start_time)

    print(str(seconds / 60) + " minutes")

    model.save_weights('models/seq2seq.h5')

    return model

def decode_sequence(input_seq, encoder_model, decoder_model, wordtoidx, idxtoword, padded_length=44, max_length=3000):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, max_length))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, wordtoidx['bos']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    formatted_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idxtoword[sampled_token_index]

        if sampled_char != 'eos' and sampled_char != 'pad':
            formatted_sentence += sampled_char + ' '
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'eos' or len(decoded_sentence) > padded_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, max_length))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
    return formatted_sentence

def test(encoder_model, decoder_model):
    x_test = np.load('x_test.npy')
    id_test = np.load('id_test.npy')

    with open('tokenizer.pickle', 'rb') as handle:
        wordtoidx = pickle.load(handle)
    idxtoword = {v: k for k, v in wordtoidx.items()}
    result = {}
    for i in range(len(x_test)):
        result[id_test[i]] = decode_sequence(x_test[i].reshape((1, 80, 4096)),
                                             encoder_model, decoder_model, wordtoidx, idxtoword)
    return result
def peer_review(encoder_model, decoder_model):
    x_peer_review = np.load('x_peer_review.npy')
    id_peer_review = np.load('id_peer_review.npy')

    with open('tokenizer.pickle', 'rb') as handle:
        wordtoidx = pickle.load(handle)
    idxtoword = {v: k for k, v in wordtoidx.items()}
    result = {}
    for i in range(len(x_peer_review)):
        result[id_peer_review[i]] = decode_sequence(x_peer_review[i].reshape((1, 80, 4096)),
                                             encoder_model, decoder_model, wordtoidx, idxtoword)
    return result
if __name__ == "__main__":
    import os
    import h5py
    from keras.models import Model, load_model
    import numpy as np
    import pickle
    from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, Concatenate, Lambda
    from keras.utils import to_categorical
    from keras.preprocessing.sequence import pad_sequences
    from keras.callbacks import ModelCheckpoint
    import keras.backend as K

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='test', type=str)
    parser.add_argument('-save_dir', default='./models')
    parser.add_argument('-output_dir', default='./output/output.txt')
    parser.add_argument('-model_dir', default='./models/seq2seq.h5')
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-epochs', default=200)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    seq2seq_model = seq2seq_model()
    seq2seq_model.summary()

    if args.mode == 'test':
        import pandas as pd
        encoder_model, decoder_model = inference_model()
        encoder_model.load_weights(args.model_dir, by_name=True)
        decoder_model.load_weights(args.model_dir, by_name=True)
        predicted = test(encoder_model, decoder_model)
        output = pd.DataFrame.from_dict(predicted, orient='index').to_csv(args.output_dir, header=None)
    elif args.mode == 'peer_review':
        import pandas as pd
        encoder_model, decoder_model = inference_model()
        encoder_model.load_weights(args.model_dir, by_name=True)
        decoder_model.load_weights(args.model_dir, by_name=True)
        predicted = peer_review(encoder_model, decoder_model)
        output = pd.DataFrame.from_dict(predicted, orient='index').to_csv(args.output_dir, header=None)
    else:
        train(seq2seq_model, batch_size=int(args.batch_size), epochs=int(args.epochs))