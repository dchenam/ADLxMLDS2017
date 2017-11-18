def S2VT_model(max_length=3000, hidden_units=1000):

    encoder_padding = Input(shape=(44, 4096), name='encoder_padding')
    decoder_padding = Input(shape=(80, 3000), name='decoder_padding')

    encoder_input = Input(shape=(80, 4096), name='encoder_input')
    decoder_input = Input(shape=(44, 3000), name='decoder_input')

    encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name='encoder_lstm', implementation=2)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name='decoder_lstm', implementation=2)

    # Encoding Stage
    encoder_output_1, lstm1_h, lstm1_c = encoder_lstm(encoder_input)
    encoder_states = [lstm1_h, lstm1_c]

    encoder_concat = Concatenate(axis=-1, name='encoding_concat')([decoder_padding, encoder_output_1])
    _, lstm2_h, lstm2_c = decoder_lstm(encoder_concat)
    decoder_states = [lstm2_h, lstm2_c]

    # Decoding Stage
    encoder_output_2, _, _ = encoder_lstm(encoder_padding, initial_state=encoder_states)
    decoder_concat = Concatenate(axis=-1, name='decoding_concat')([decoder_input, encoder_output_2])
    decoder_output, _, _ = decoder_lstm(decoder_concat, initial_state=decoder_states)
    decoder_dense = Dense(max_length, activation='softmax', name='decoder_dense')
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_padding, decoder_padding, encoder_input, decoder_input], decoder_output)

    return model

def inference_model(max_length=3000, hidden_units=1000):
    encoder_padding = Input(shape=(1, 4096), name='encoder_padding')
    decoder_padding = Input(shape=(1, 3000), name='decoder_padding')

    encoder_input = Input(shape=(80, 4096), name='encoder_input')
    decoder_input = Input(shape=(1, 3000), name='decoder_input')

    encoder_lstm = LSTM(hidden_units, return_state=True, name='encoder_lstm', implementation=2)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name='decoder_lstm', implementation=2)

    # Encoding Stage
    encoder_output_1, lstm1_h, lstm1_c = encoder_lstm(encoder_input)
    encoder_states = [lstm1_h, lstm1_c]
    repeat_input_1 = RepeatVector(1, name='repeat_encoder_output')(encoder_output_1)
    encoder_concat = Concatenate(axis=-1, name='encoding_concat')([decoder_padding, repeat_input_1])
    _, lstm2_h, lstm2_c = decoder_lstm(encoder_concat)
    decoder_states = [lstm2_h, lstm2_c]

    encoder_model = Model([decoder_padding, encoder_input], decoder_states)

    # Decoding Stage
    encoder_output_2, _, _ = encoder_lstm(encoder_padding, initial_state=encoder_states)
    repeat_input_2 = RepeatVector(1, name='repeat_encoder_output2')(encoder_output_2)
    decoder_concat = Concatenate(axis=-1, name='decoding_concat')([decoder_input, repeat_input_2])

    decoder_dense = Dense(max_length, activation='softmax', name='decoder_dense')

    decoder_state_input_h = Input(shape=(hidden_units,))
    decoder_state_input_c = Input(shape=(hidden_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_output, state_h, state_c = decoder_lstm(
        decoder_concat, initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]
    decoder_output = decoder_dense(decoder_output)

    decoder_model = Model(
        [encoder_padding, encoder_input, decoder_input] + decoder_states_inputs,
        [decoder_output] + decoder_states)

    return encoder_model, decoder_model

def train(model, data, batch_size, epochs):
    max_length = 3000
    padded_length = 44

    x_train, y_train,y_train_shift,  x_test, y_test, y_test_shift = data
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    def data_generator(features, labels, shifted_labels, batch_size):
        batch_encoder_input = np.zeros((batch_size, features.shape[1], features.shape[2]))
        batch_decoder_input = np.zeros((batch_size, padded_length, max_length))
        batch_decoder_output = np.zeros((batch_size, padded_length, max_length))
        x_train_size = len(features)
        sample_index = 0
        while True:
            for i in range(batch_size):
                caption_index = np.random.randint(len(labels[sample_index]))
                batch_encoder_input[i] = features[sample_index]
                batch_decoder_input[i] = to_categorical(shifted_labels[sample_index][caption_index], 3000)
                batch_decoder_output[i] = to_categorical(labels[sample_index][caption_index], 3000)

                sample_index += 1

                if sample_index >= x_train_size:
                    sample_index = 0
            batch_encoder_padding = np.zeros((batch_size, 44, 4096))
            batch_decoder_padding = np.zeros((batch_size, 80, 3000))
            yield [batch_encoder_padding, batch_decoder_padding,
                   batch_encoder_input, batch_decoder_input], batch_decoder_output

    import time
    start_time = time.time()

    model.fit_generator(
        data_generator(x_train, y_train, y_train_shift, batch_size=batch_size),
        steps_per_epoch=1450/batch_size,
        epochs=epochs,
        validation_data=data_generator(x_test, y_test, y_test_shift, batch_size=batch_size),
        validation_steps=1450/batch_size,
        verbose=1)
    end_time = time.time()
    seconds = (end_time - start_time)

    print(str(seconds / 60) + " minutes")

    model.save_weights('models/S2VT.h5')

    return model

def decode_sequence(input_seq, encoder_model, decoder_model, wordtoidx, idxtoword, padded_length=44, max_length=3000):
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, max_length))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, wordtoidx['bos']] = 1.

    encoder_padding = np.zeros((1, 44, 4096))
    decoder_padding = np.zeros((1, 80, 3000))

    # Encode the input as state vectors.
    decoder_states = encoder_model.predict([decoder_padding, input_seq])

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    formatted_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [encoder_padding, input_seq, target_seq] + decoder_states)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idxtoword[sampled_token_index]

        if sampled_char != 'eos':
            formatted_sentence += sampled_char + ' '
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'eos' or len(decoded_sentence) > padded_length):
            stop_condition = True

        print(sampled_token_index)

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, max_length))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
    return formatted_sentence

def test(encoder_model, decoder_model):
    x_test = np.load('x_test.npy')
    idx_test = np.load('idx_test.npy')

    with open('word_to_idx.pickle', 'rb') as handle:
        wordtoidx = pickle.load(handle)
    with open('idx_to_word.pickle', 'rb') as handle:
        idxtoword = pickle.load(handle)

    result = {}
    for i in range(len(x_test)):
        result[idx_test[i]] = decode_sequence(x_test[i].reshape((1, 80, 4096)), encoder_model, decoder_model, wordtoidx, idxtoword)

    return result

if __name__ == "__main__":
    import os
    import h5py
    from keras.models import Model, load_model
    import numpy as np
    import pickle
    from keras.layers import Input, LSTM, Dense, Masking, RepeatVector, Concatenate, Lambda
    from keras.utils import to_categorical
    from keras.callbacks import ModelCheckpoint
    import keras.backend as K

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='test', type=str)
    parser.add_argument('--save_dir', default='./models')
    parser.add_argument('--output_dir', default='./result.txt')
    parser.add_argument('--model_dir', default='./models/S2VT.h5')
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-epochs', default=200)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    S2VT_model = S2VT_model()
    S2VT_model.summary()

    if args.mode == 'test':
        import pandas as pd
        encoder_model, decoder_model = inference_model()
        encoder_model.load_weights(args.model_dir, by_name=True)
        decoder_model.load_weights(args.model_dir, by_name=True)
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
        train(S2VT_model, data, batch_size=int(args.batch_size), epochs=int(args.epochs))