from numpy import squeeze as np_squeeze
import matplotlib.pyplot as plt
from textwrap import wrap


def get_predict(test_loader,
                encoder,
                decoder,
                device,
                vocab,
                shwetank=False,
                mode='sample',
                beam_size: int = 3,
                max_seq_len: int = 12):
    if mode not in ('sample', 'beam', 'nucleus'):
        raise ValueError('mode must be sample, beam, or nucleus')

    orig_image, image = next(iter(test_loader))
    features = encoder(image.to(device)).unsqueeze(1)

    if mode == 'beam':
        if shwetank:
            output = decoder.shwetank_beam_search(features, beam_size, max_seq_len)
        else:
            output = decoder.nic_beam_search(features, beam_size, max_seq_len)

        print('beam top %s:' % beam_size, output)
        caption = output[0][0]
    elif mode == 'nucleus':
        if shwetank:
            output = decoder.shwetank_nucleus(features, max_seq_len)
            beam_output = decoder.shwetank_beam_search(features, beam_size, max_seq_len)
            sample_output = decoder.shwetank_sample(features, max_len=max_seq_len)
        else:
            output = decoder.nic_nucleus(features, max_seq_len)
            beam_output = decoder.nic_beam_search(features, beam_size, max_seq_len)
            sample_output = decoder.nic_sample(features, max_len=max_seq_len)

        caption = ' '.join([vocab.ind2word[i] for i in output])

        print('sample:', ' '.join([vocab.ind2word[i] for i in sample_output]))
        print('beam:', beam_output[0][0])
        print('nucleus:', caption)
    else:
        if shwetank:
            output = decoder.shwetank_sample(features, max_len=max_seq_len)
        else:
            output = decoder.nic_sample(features, max_len=max_seq_len)

        caption = ' '.join([vocab.ind2word[i] for i in output])

    caption = "\n".join(wrap(caption, 60))
    plt.imshow(np_squeeze(orig_image))
    plt.title(caption)
    plt.show()
