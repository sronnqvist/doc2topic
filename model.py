from keras.models import Model
from keras.layers import Input, Embedding, dot, Reshape, Activation
from keras.regularizers import l1
from keras.optimizers import Adam
from measures import *

def init_model(data_len, vocab_len, emb_dim, l1_doc, l1_word, lr):
    inlayerD = Input((1,))
    inlayerW = Input((1,))
    EmbD = Embedding(data_len, emb_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_doc), name="docvecs")
    EmbW = Embedding(vocab_len, emb_dim, input_length=1, trainable=True, activity_regularizer=l1(l1_word), name="wordvecs")

    embD = EmbD(inlayerD)
    embDa = Activation('relu')(embD)
    embD = Reshape((emb_dim, 1))(embDa)

    embW = EmbW(inlayerW)
    embWa = Activation('relu')(embW)
    embW = Reshape((emb_dim, 1))(embWa)

    #sim = dot([embD, embW], 0, normalize=True)
    dot_prod = dot([embD, embW], 1, normalize=False)
    dot_prod = Reshape((1,))(dot_prod)

    output = Activation('sigmoid')(dot_prod)

    opt = Adam(lr=lr, amsgrad=True)

    model = Model(inputs=[inlayerD,inlayerW], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', fmeasure, precision])
    
    return model
