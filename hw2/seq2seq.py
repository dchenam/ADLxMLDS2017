def seq2seq_model(max_length=3000, padded_length=44, hidden_units=256):
    encoder_input = Input(shape=(80, 4096))
    masked_encoder_input = Masking()(encoder_input)
    encoder = LSTM(hidden_units, return_state=True)
    encoder_output, state_h, state_c = encoder(masked_encoder_input)

    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(padded_length, max_length))
    masked_decoder_input = Masking()(decoder_input)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(masked_decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(max_length, activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)

    # ----Decoding Inference Model------#
    encoder_model = Model(encoder_input, encoder_states)

    decoder_state_input_h = Input(shape=(hidden_units,))
    decoder_state_input_c = Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_output, state_h, state_c = decoder_lstm(
        masked_decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_output = decoder_dense(decoder_output)
    decoder_model = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_output] + decoder_states)

    return model, encoder_model, decoder_model

def train(model, data, max_length=3000, padded_length=44, batch_size=32, epochs=200):

    seq2seq_model, encoder_model, decoder_model = model
    x_train, y_train, y_train_shift, x_test, y_test, y_test_shift = data

    seq2seq_model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

    def data_generator(features, labels, shifted_labels, batch_size):
        batch_encoder_input = np.zeros((batch_size, features.shape[1], features.shape[2]))
        batch_decoder_input = np.zeros((batch_size, padded_length, max_length))
        batch_decoder_output = np.zeros((batch_size, padded_length, max_length))
        x_train_size = len(features)
        sample_index = 0
        caption_index = 0
        while True:
            for i in range(batch_size):

                batch_encoder_input[i] = features[sample_index]
                batch_decoder_input[i] = to_categorical(shifted_labels[sample_index][caption_index], 3000)
                batch_decoder_output[i] = to_categorical(labels[sample_index][caption_index], 3000)

                caption_index += 1

                if caption_index > len(labels[sample_index]) - 1:
                    sample_index += 1
                    caption_index = 0

                if (sample_index > x_train_size - 1):
                    sample_index = 0
                    caption_index = 0

            yield [batch_encoder_input, batch_decoder_input], batch_decoder_output

    import time
    start_time = time.time()
    seq2seq_model.fit_generator(
        data_generator(x_train, y_train, y_train_shift, batch_size=batch_size),
        steps_per_epoch=len(x_train) / batch_size,
        epochs=epochs,
        validation_data=data_generator(x_test, y_test, y_test_shift, batch_size=batch_size),
        validation_steps=len(y_test) / batch_size,
        verbose=1)
    end_time = time.time()
    seconds = (end_time - start_time)

    print(str(seconds / 60) + " minutes")

    seq2seq_model.save('models/seq2seq.h5')
    encoder_model.save('models/encoder.h5')
    decoder_model.save('models/decoder.h5')
    return seq2seq_model

def decode_sequence(input_seq, wordtoidx, idxtoword, encoder_model, decoder_model, padded_length=44, max_length=3000):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, padded_length, max_length))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, wordtoidx['bos']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idxtoword[sampled_token_index]

        if sampled_char == 'eos':
            print('reached: eos')
            decoded_sentence += '.'
        else:
            decoded_sentence += sampled_char + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'eos' or len(decoded_sentence) > padded_length):
            stop_condition = True

        print(sampled_token_index)
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, padded_length, max_length))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def test(encoder_model, decoder_model):

    x_test = np.load('x_test.npy')
    idx_test = np.load('idx_test.npy')

    with open('word_to_idx.pickle', 'rb') as handle:
        wordtoidx = pickle.load(handle)
    with open('idx_to_word.pickle', 'rb') as handle:
        idxtoword = pickle.load(handle)

    result = {}

    for i in range(len(x_test)):
        result[idx_test[i]] = decode_sequence(x_test[i].reshape((1, 80, 4096)), wordtoidx, idxtoword, encoder_model, decoder_model)

    return result

if __name__ == "__main__":

    import os
    import h5py
    from keras.models import load_model
    import numpy as np
    import pickle
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Masking
    from keras.utils import to_categorical
    from keras.preprocessing.sequence import pad_sequences

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--save_dir', default='./models')
    parser.add_argument('--output_dir', default='./result.txt')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    seq2seq_model, encoder_model, decoder_model = seq2seq_model()

    seq2seq_model.summary()

    model = [seq2seq_model, encoder_model, decoder_model]
    if args.mode == 'test':
        import pandas as pd

        model = load_model('models/seq2seq.h5')
        encoder_model = load_model('models/encoder.h5')
        decoder_model = load_model('models/decoder.h5')

        predicted = test(encoder_model, decoder_model)
        output = pd.DataFrame.from_dict(predicted, orient='index').to_csv(args.output_dir, header=None)
    else:
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        y_train_shift = np.load('y_shift_train.npy')
        x_test = np.load('x_test.npy')
        y_test = np.load('y_test.npy')
        y_test_shift = np.load('y_shift_test.npy')
        data = [x_train, y_train, y_train_shift, x_test, y_test, y_test_shift]

        train(model, data)