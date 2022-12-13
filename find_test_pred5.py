from dataloader import CoCoLoader
from transforms import text_trans, aug_img_trans, aug_img_trans2, img_trans
from architect import Encoder, Decoder
from os.path import join, dirname, exists
from os import makedirs
from predict import get_predict

import torch

# Requires image and annotations from https://cocodataset.org/#download

test_img_dir = r'C:\Users\kevin\Desktop\NEU_Cloud\202209_Fall_Semester\01_Deep_Learning\04_Project\01_NIC\cocoapi\images\test2017'
train_ann_fp = r'C:\Users\kevin\Desktop\NEU_Cloud\202209_Fall_Semester\01_Deep_Learning\04_Project\01_NIC\cocoapi\annotations\captions_train2017.json'
test_ann_fp = r'C:\Users\kevin\Desktop\NEU_Cloud\202209_Fall_Semester\01_Deep_Learning\04_Project\01_NIC\cocoapi\annotations\image_info_test2017.json'

file_dir = dirname(__file__)
model_dir = join(join(file_dir, 'models'), '05_train5_py_models')


if __name__ == '__main__':
    embed_size = 300
    lstm_dim = 128
    batch_size = 1
    vocab_thresh = 8

    if not exists(model_dir):
        makedirs(model_dir)

    test_loader = CoCoLoader(test_img_dir,
                             test_ann_fp,
                             vocab_fp=train_ann_fp,
                             batch_size=batch_size,
                             img_transform=img_trans,
                             target_transform=text_trans,
                             vocab_threshold=vocab_thresh,
                             load_captions=False)

    vocab = test_loader.vocab
    vocab_size = len(vocab)
    encoder = Encoder(embed_size)
    encoder.eval()
    decoder = Decoder(embed_size,
                      lstm_dim,
                      vocab_size,
                      vocab.word2ind[vocab.start_word],
                      vocab.word2ind[vocab.unknown_word],
                      vocab.word2ind[vocab.end_word],
                      vocab.ind2word,
                      prob_mass=0.15)  # prob_mass somewhere between 10-20
    decoder.eval()
    decoder.load_state_dict(torch.load(join(model_dir, 'decoder-1.pkl')))
    encoder.load_state_dict(torch.load(join(model_dir, 'encoder-1.pkl')), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    get_predict(test_loader,
                encoder,
                decoder,
                device,
                vocab,
                shwetank=True,
                mode='nucleus',
                beam_size=20,
                max_seq_len=13)
