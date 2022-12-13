from abc import ABC
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from json import load as json_load
from PIL.Image import open as pil_img_open
from torchtext.data.utils import get_tokenizer
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from vocabulary import Vocabulary
from collections import defaultdict
from os.path import join
from numpy.random import choice
from numpy import where as np_where, arange as np_arange, array as np_array
from albumentations import Compose
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
from numpy import squeeze as np_squeeze
import albumentations as A
from albumentations.pytorch import ToTensorV2
from textwrap import wrap

import cv2
from time import time

# spacy_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
spacy_tokenizer = get_tokenizer("basic_english")


class CoCoLoader(object):
    def __init__(self,
                 img_dir: str,
                 annotations_fp: str,
                 vocab_fp: str = None,
                 batch_size: int = 1,
                 vocab_threshold: int = -1,
                 reload: bool = False,
                 start_word: str = '<start>',
                 end_word: str = '<end>',
                 unknown_word: str = '<unknown>',
                 img_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 both_transform: Optional[Callable] = None,
                 word_to_ind: bool = True,
                 image_to_ann: bool = False,
                 load_captions: bool = False,
                 num_workers=0):

        coco_data = CoCoDataset(img_dir=img_dir,
                                annotations_fp=annotations_fp,
                                vocab_fp=vocab_fp,
                                batch_size=batch_size,
                                vocab_threshold=vocab_threshold,
                                reload=reload,
                                start_word=start_word,
                                end_word=end_word,
                                unknown_word=unknown_word,
                                img_transform=img_transform,
                                target_transform=target_transform,
                                both_transform=both_transform,
                                word_to_ind=word_to_ind,
                                image_to_ann=image_to_ann,
                                load_captions=load_captions)
        inds = coco_data.get_batch_ind()
        rand_samp = SubsetRandomSampler(indices=inds)
        self.vocab = coco_data.vocabulary
        self.num_of_iter = len(coco_data.ids) / batch_size
        self.batch_size = batch_size
        self.loader = DataLoader(dataset=coco_data,
                                 num_workers=num_workers,
                                 batch_sampler=BatchSampler(sampler=rand_samp,
                                                            batch_size=batch_size,
                                                            drop_last=False))

    def __iter__(self):
        curr_iter = 0

        while curr_iter < self.num_of_iter:
            ind = self.loader.dataset.get_batch_ind()
            self.loader.batch_sampler.sampler = SubsetRandomSampler(indices=ind)
            yield next(iter(self.loader))
            curr_iter += 1

    def __len__(self):
        return self.num_of_iter


class CoCoDataset(VisionDataset, ABC):
    def __init__(self,
                 img_dir: str,
                 annotations_fp: str,
                 vocab_fp: str = None,
                 batch_size: int = 1,
                 vocab_threshold: int = -1,
                 reload: bool = False,
                 start_word: str = '<start>',
                 end_word: str = '<end>',
                 unknown_word: str = '<unknown>',
                 img_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 both_transform: Optional[Callable] = None,
                 word_to_ind: bool = True,
                 image_to_ann: bool = False,
                 load_captions: bool = False
                 ):
        super().__init__(img_dir, both_transform, img_transform, target_transform)
        self.__img_fns = dict()
        self.__captions = dict()
        self.load_captions = load_captions
        self.is_alb = True if isinstance(img_transform, Compose) else False
        self.img_dir = img_dir

        if vocab_fp:
            self.vocabulary = Vocabulary(vocab_fp,
                                         vocab_threshold=vocab_threshold,
                                         reload=reload,
                                         start_word=start_word,
                                         end_word=end_word,
                                         unknown_word=unknown_word)
        else:
            self.vocabulary = Vocabulary(annotations_fp,
                                         vocab_threshold=vocab_threshold,
                                         reload=reload,
                                         start_word=start_word,
                                         end_word=end_word,
                                         unknown_word=unknown_word)

        self.image_to_ann = image_to_ann
        self.batch_size = batch_size
        self.word_to_ind = word_to_ind
        self.__load_annotations(annotations_fp)

        if self.load_captions:
            self.ids = list(sorted(self.__captions.keys()))
            self.captions_lens = [len(self.vocabulary.captions[id]) for id in self.ids]
        else:
            self.ids = list(sorted(self.__img_fns.keys()))
            self.captions_lens = list()

    def __load_annotations(self,
                           annotations_fp: str):
        if self.image_to_ann:
            self.__captions = defaultdict(list)

        with open(annotations_fp, 'r') as f:
            data = json_load(f)
            assert type(data) == dict,\
                'annotation file format {} is not supported'.format(type(data))

        if 'images' in data:
            for image in data['images']:
                self.__img_fns[image['id']] = image['file_name']

        if self.load_captions and 'annotations' in data:
            for annotations in data['annotations']:
                if self.image_to_ann:
                    self.__captions[annotations['image_id']].append(annotations)
                else:
                    self.__captions[annotations['id']] = annotations

    def __getitem__(self,
                    ind: int) -> (Tuple[Any, Any], Any):
        id_ind = self.ids[ind]

        if self.load_captions:
            img_id = self.__captions[id_ind]['image_id']
            img_fn = self.__img_fns[img_id]

            if self.is_alb:
                orig_img_data = cv2.imread(join(self.img_dir,
                                                img_fn))
                orig_img_data = cv2.cvtColor(orig_img_data, cv2.COLOR_BGR2RGB)
            else:
                orig_img_data = pil_img_open(join(self.img_dir,
                                                  img_fn)).convert('RGB')

            if not self.image_to_ann and self.word_to_ind:
                caption = list()
                caption.append(self.vocabulary(self.vocabulary.start_word))
                caption.extend([self.vocabulary(word) for word in self.vocabulary.captions[id_ind]])
                caption.append(self.vocabulary(self.vocabulary.end_word))
            else:
                caption = self.__captions[id_ind]['caption']

            '''
            # Show Original Image with Caption
            text_caption = "\n".join(wrap(' '.join(self.__captions[id_ind]['caption']), 60))
            plt.imshow(np_squeeze(orig_img_data))
            plt.title('Original - %s' % text_caption)
            plt.show()
            '''
        else:
            img_fn = self.__img_fns[id_ind]

            if self.is_alb:
                orig_img_data = cv2.imread(join(self.img_dir,
                                                img_fn))
                orig_img_data = cv2.cvtColor(orig_img_data, cv2.COLOR_BGR2RGB)
            else:
                orig_img_data = pil_img_open(join(self.img_dir,
                                                  img_fn)).convert('RGB')

            caption = None
            text_caption = None

        if self.transforms is not None:
            if self.is_alb:
                img_data = self.transforms.transform(image=orig_img_data)['image'].T

                '''
                # Show Augmented Image with Caption
                self.transforms.transform = A.Compose([t for t in self.transforms.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
                plt.imshow(np_squeeze(self.transforms.transform(image=orig_img_data)['image']))
                plt.title('Augmented - %s' % text_caption)
                plt.show()
                '''
            else:
                img_data = self.transforms.transform(orig_img_data)

            if caption:
                caption = self.transforms.target_transform(caption)
        else:
            img_data = orig_img_data

        if self.load_captions:
            return img_data, caption
        else:
            return np_array(orig_img_data), img_data

    def get_batch_ind(self):
        if self.image_to_ann or not self.word_to_ind or len(self.captions_lens) < 1:
            all_ind = list(range(len(self.ids)))
            return list(choice(all_ind, size=self.batch_size))
        else:
            pick_cap_len = choice(self.captions_lens)
            pick_all_ind = np_where([self.captions_lens[k] == pick_cap_len
                                     for k in np_arange(len(self.captions_lens))])[0]
            ind_batch = list(choice(pick_all_ind, size=self.batch_size))
            return ind_batch

    def __len__(self) -> int:
        return len(self.ids)
