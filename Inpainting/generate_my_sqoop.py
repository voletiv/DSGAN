# import argparse
# import h5py
import io
import math
import numpy as np
import os
import random
import string
import sys
import time
import tqdm

from functools import partial
from PIL import Image, ImageDraw, ImageFont

RELATIONS = ['left_of', 'right_of', 'above', 'below']
COLORS = ['red', 'green', 'blue', 'yellow', 'cyan', 'purple', 'brown', 'gray']
SHAPES = list(string.ascii_uppercase) + ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']


class Object(object):
    def __init__(self, fontsize, angle=0, pos=None, shape=None):
        self.font = FONT_OBJECTS[fontsize]
        width, self.size = self.font.getsize('A')
        self.angle = angle
        angle_rad = angle / 180 * math.pi
        self.rotated_size =  math.ceil(self.size * (abs(math.sin(angle_rad)) + abs(math.cos(angle_rad))))
        self.pos = pos
        self.shape = shape

    def overlap(self, other):
        min_dist = (self.rotated_size + other.rotated_size) // 2 + 1
        return (abs(self.pos[0] - other.pos[0]) < min_dist and
                abs(self.pos[1] - other.pos[1]) < min_dist)

    def relate(self, rel, other):
        if rel == 'left_of':
            return self.pos[0] < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0]
        if rel == 'above':
            return self.pos[1] > other.pos[1]
        if rel == 'below':
            return self.pos[1] < other.pos[1]
        raise ValueError(rel)

    def draw(self):
        obj_img = Image.new('RGBA', (self.size, self.size))
        draw = ImageDraw.Draw(obj_img)
        draw.text((0,0), self.shape, font=self.font, fill='green')

        #if self.angle != 0:
        #  img = img.rotate(self.angle, expand=True, resample=Image.LINEAR)

        return obj_img


def draw_scene(objects, image_size=64):
    img = Image.new('RGB', (image_size, image_size))
    for obj in objects:
        obj_img = obj.draw()
        obj_pos = (obj.pos[0] - obj_img.size[0] // 2, image_size - obj.pos[1] - obj_img.size[1] // 2 )
        img.paste(obj_img, obj_pos, obj_img)

    return img

    def relate(self, rel, other):
        if rel == 'left_of':
            return self.pos[0] < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0]
        if rel == 'above':
            return self.pos[1] > other.pos[1]
        if rel == 'below':
            return self.pos[1] < other.pos[1]
        raise ValueError(rel)


class Sampler:
    def __init__(self, test, seed, objects):
        self._test = test
        self._rng = np.random.RandomState(seed)
        self.objects = objects

    def _choose(self, list_like, num=1):
        idx = self._rng.randint(0, len(list_like), num)
        return [list_like[i] for i in idx]

    def _rejection_sample(self, restricted=[], num=1):
        objs = list(self.objects)
        for obj in restricted:
            objs.remove(obj)
        rand_objects = list(self._rng.choice(objs, num))
        return rand_objects

    def sample_relation(self, num=1):
        return self._choose(RELATIONS, num)

    def sample_object(self, restricted=[]):
        return self._rejection_sample(restricted)


class _LongTailSampler(Sampler):
    def __init__(self, dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_probs = dist

    def sample_object(self, restricted=[], num=1):
        if self._test:
            return self._rejection_sample(restricted=restricted, num=num)
        else:
            return self._rejection_sample(self.object_probs, restricted=restricted, num=num)

    def _rejection_sample(self, shape_probs=None, restricted=[], num=1):
        dist = np.array(self.object_probs)
        for obj in restricted:
            dist[self.objects.index(obj)] = 0
        dist = dist/dist.sum()
        return self._rng.choice(self.objects, num, p=dist)


def LongTailSampler(long_tail_dist):
    return partial(_LongTailSampler, long_tail_dist)


def generate_scene(rng, sampler, objects=[], restrict=False, num_objects=5):
    orig_objects = objects

    objects = list(orig_objects)
    place_failures = 0

    if restrict:
        restricted_obj = [obj.shape for obj in orig_objects]
    else:
        restricted_obj = []

    # first, select which object to draw by rejection sampling
    if len(objects) < num_objects:
        shapes = sampler.sample_object(restricted_obj, num_objects-len(objects))

    while len(objects) < num_objects:
        # print("generate_scene", len(objects)+1)
        new_object = get_random_spot(rng, objects)
        if new_object is None:
            place_failures += 1
            if place_failures == 10:
                # reset generation
                objects = list(orig_objects)
                place_failures = 0
            continue

        objects.append(new_object)

    for i in range(2, num_objects):
        objects[i].shape = shapes[i-2]

    return objects


def get_random_spot(rng, objects, rel=None, rel_holds=True, rel_obj=0,
                    min_obj_size=10, max_obj_size=15, rotate=True, image_size=64):
    """Get a spot for a new object that does not overlap with existing ones."""
    # then, select the object size

    pos_found = False

    for i in range(10):
        size = rng.randint(min_obj_size, max_obj_size + 1)
        angle = rng.randint(0, 360) if rotate else 0
        obj = Object(size, angle)
        min_center = obj.rotated_size // 2 + 1
        max_center = image_size - obj.rotated_size // 2 - 1

        if rel is not None:
            if rel_holds == False:
                # do not want the relation to be true
                max_center_x = objects[rel_obj].pos[0] if rel == 'left_of' else max_center
                min_center_x = objects[rel_obj].pos[0] if rel == 'right_of' else min_center
                max_center_y = objects[rel_obj].pos[1] if rel == 'below' else max_center
                min_center_y = objects[rel_obj].pos[1] if rel == 'above' else min_center
            else:
                # want the relation to be true
                min_center_x = objects[rel_obj].pos[0] if rel == 'left_of' else min_center
                max_center_x = objects[rel_obj].pos[0] if rel == 'right_of' else max_center
                min_center_y = objects[rel_obj].pos[1] if rel == 'below' else min_center
                max_center_y = objects[rel_obj].pos[1] if rel == 'above' else max_center

            if min_center_x >= max_center_x: continue
            if min_center_y >= max_center_y: continue

        else:
            min_center_x = min_center_y = min_center
            max_center_x = max_center_y = max_center

        x = rng.randint(min_center_x, max_center_x)
        y = rng.randint(min_center_y, max_center_y)
        obj.pos = (x, y)
        # make sure there is no overlap between bounding squares
        if (any([abs(obj.pos[0] - other.pos[0]) < 5 for other in objects]) or
            any([abs(obj.pos[1] - other.pos[1]) < 5 for other in objects])):
            continue
        if any([obj.overlap(other) for other in objects]):
            continue
        return obj

    return None


def generate_scene_and_question(pair, sampler, rng, label, rel, num_objects, vocab):
    # x rel y has value label where pair == (x, y)
    x, y = pair
    question = [x, rel, y]
    q_oh = [vocab.index(x), RELATIONS.index(rel), vocab.index(y)]
    gen = False
    while not gen:
        obj1 = get_random_spot(rng, [])
        # print("generate_scene_and_question 2")
        obj2 = get_random_spot(rng, [obj1], rel=rel, rel_holds=label)
        if not obj2 or obj1.relate(rel, obj2) != label: continue
        obj1.shape = x
        obj2.shape = y
        # print("generating scene")
        scene = generate_scene(rng, sampler, objects=[obj1, obj2], restrict=True, num_objects=num_objects)
        gen = True

    return scene, question, q_oh


def gen_image_and_condition(obj_pairs, sampler, seed, num_objects, vocab):
    presampled_relations = sampler.sample_relation(num=len(obj_pairs))      # pre-sample relations

    rng = np.random.RandomState(seed)
    before = time.time()
    images = []
    qs = []
    qs_oh = []
    labels = []
    for i in tqdm.tqdm(range(len(obj_pairs))):
        # label = (i % 2) == 0
        label = True
        labels.append(label)
        scene, question, q_oh = generate_scene_and_question(obj_pairs[i], sampler, rng, label, presampled_relations[i], num_objects, vocab)
        # buffer_ = io.BytesIO()
        image = np.array(draw_scene(scene[:num_objects]))
        # image.save(buffer_, format='png')
        # buffer_.seek(0)
        # image = np.frombuffer(buffer_.read(), dtype='uint8')
        images.append(image)
        qs.append(question)
        qs_oh.append(q_oh)

    return images, qs, qs_oh, labels


def gen_my_sqoop(vocab=None, num_objects=5, pairings_per_obj=0, num_repeats=10,
                    val=False, test=False, num_repeats_eval=10):

    if vocab is None:
        vocab = SHAPES

    if pairings_per_obj == 0:
        pairings_per_obj = len(vocab)-1

    uniform_dist = [1.0 / len(vocab) ]*len(vocab)
    sampler_class = LongTailSampler(uniform_dist)

    train_sampler = sampler_class(False, 1, vocab)
    val_sampler   = sampler_class(True,  2, vocab)
    test_sampler  = sampler_class(True,  3, vocab)

    train_pairs = []
    val_pairs   = []
    test_pairs  = []

    all_pairs = [(x,y) for x in vocab for y in vocab if x != y]
    chosen = list(all_pairs)
    for i, x in enumerate(vocab):
        ys = sorted(random.sample(vocab[:i] + vocab[i+1:], pairings_per_obj))
        for y in ys:
            chosen.remove((x,y))
            train_pairs += [(x,y)]*num_repeats

    print("Generating train images & qs")
    train_ims, train_qs, train_qs_oh, train_labels = gen_image_and_condition(train_pairs, train_sampler, 1, num_objects, vocab)

    if val or test:
        remaining = list(chosen)
        print('number of zero shot pairs: %d' % len(remaining))
        # dev / test pairs are all unseen
        if test:
            val_slice = len(remaining) // 2
        else:
            val_slice = len(remaining)

        for pair in remaining[:val_slice]:
            val_pairs  += [pair] * num_repeats_eval
        if test:
            for pair in remaining[val_slice:]:
                test_pairs += [pair] * num_repeats_eval

    if val:
        print("Generating val images & qs")
        val_ims, val_qs, val_qs_oh, val_labels = gen_image_and_condition(val_pairs, val_sampler, 2, num_objects, vocab)

    if test:
        print("Generating test images & qs")
        test_ims, test_qs, test_qs_oh, test_labels = gen_image_and_condition(test_pairs, test_sampler, 3, num_objects, vocab)

    if not val and not test:
        return train_ims, train_qs, train_qs_oh, train_labels
    elif not val and test:
        return train_ims, train_qs, train_qs_oh, train_labels, test_ims, test_qs, test_qs_oh, test_labels
    elif val and not test:
        return train_ims, train_qs, train_qs_oh, train_labels, val_ims, val_qs, val_qs_oh, val_labels
    elif val and test:
        return train_ims, train_qs, train_qs_oh, train_labels, val_ims, val_qs, val_qs_oh, val_labels, test_ims, test_qs, test_qs_oh, test_labels


# parser = argparse.ArgumentParser()
# parser.add_argument('--program', type=str, default='best',
#                     choices=('best', 'noand', 'chain', 'chain2', 'chain3', 'chain_shortcut'))
# parser.add_argument('--num_shapes', type=int, default=len(SHAPES))
# parser.add_argument('--num_colors', type=int, default=1)
# parser.add_argument('--num_objects', type=int, default=5)
# parser.add_argument('--rhs_variety', type=int, default=len(SHAPES) // 2)
# parser.add_argument('--split', type=str, default='systematic', choices=('systematic', 'vanilla'))
# parser.add_argument('--num_repeats', type=int, default=10)
# parser.add_argument('--num_repeats_eval', type=int, default=10)
# parser.add_argument('--data_dir', type=str, default='.')
# parser.add_argument('--mode', type=str, default='sqoop',
#                     choices=['sqoop', 'sqoop_easy_test'],
#                     help='in sqoop_easy_test mode the script generates a test set with the same '
#                             'questions as the dataset in the current directory, '
#                             'but with different images')
# parser.add_argument('--image_size', type=int, default=64)
# parser.add_argument('--min_obj_size', type=int, default=10)
# parser.add_argument('--max_obj_size', type=int, default=15)
# parser.add_argument('--no_rotate', action='store_false', dest='rotate')
# parser.add_argument('--font', default='arial.ttf')
# args = parser.parse_args()

# FONT_OBJECTS = {font_size : ImageFont.truetype(args.font) for font_size in range(10, 16) }
FONT_OBJECTS = {font_size : ImageFont.truetype('arial.ttf') for font_size in range(10, 16) }


if __name__ == '__main__':
    args.level = 'relations'
    data_full_dir = "%s/sqoop-variety_%d-repeats_%d" %(args.data_dir, args.rhs_variety, args.num_repeats)
    if args.split == 'vanilla':
        data_full_dir += "_vanilla"
    if not os.path.exists(data_full_dir):
        os.makedirs(data_full_dir)

    os.chdir(data_full_dir)
    with open('args.txt', 'w') as dst:
        print(args, file=dst)

    vocab = SHAPES[:args.num_shapes]
    if args.mode == 'sqoop':
        gen_sqoop(vocab)
    else:
        gen_image_understanding_test(vocab)