import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# thingseeg2 preprocessing
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-avg', '--average', help='Number of averages', default=80)
args = parser.parse_args()
average=int(args.average)
if average != 80:
    param = f'{average}'
else:
    param = ''

for sub in range(1, 11):
    data_dir = f'data/thingseeg2_preproc/sub-{sub:02d}/'

    if average == 80:
        eeg_data_train = np.load(data_dir + 'preprocessed_eeg_training.npy', allow_pickle=True).item()
        print(f'\nTraining EEG data shape for sub-{sub:02d}:')
        print(eeg_data_train['preprocessed_eeg_data'].shape)
        print('(Training image conditions × Training EEG repetitions × EEG channels × '
            'EEG time points)')
        train_thingseeg2 = eeg_data_train['preprocessed_eeg_data'][:,:,:,20:]
        train_thingseeg2_avg = eeg_data_train['preprocessed_eeg_data'].mean(1)[:,:,20:]
        train_thingseeg2_avg_null = eeg_data_train['preprocessed_eeg_data'].mean(1)[:,:,:20]
        np.save(data_dir + 'train_thingseeg2.npy', train_thingseeg2)
        np.save(data_dir + 'train_thingseeg2_avg.npy', train_thingseeg2_avg)
        np.save(data_dir + 'train_thingseeg2_avg_null.npy', train_thingseeg2_avg_null)
    
    eeg_data_test = np.load(data_dir + 'preprocessed_eeg_test.npy', allow_pickle=True).item()
    print(f'\nTest EEG data shape for sub-{sub:02d}:')
    print(eeg_data_test['preprocessed_eeg_data'].shape)
    print('(Test image conditions × Test EEG repetitions × EEG channels × '
        'EEG time points)')
    test_thingseeg2 = eeg_data_test['preprocessed_eeg_data'][:,:,:,20:]
    test_thingseeg2_avg = eeg_data_test['preprocessed_eeg_data'][:,:average].mean(1)[:,:,20:]
    test_thingseeg2_avg_null = eeg_data_test['preprocessed_eeg_data'][:,:average].mean(1)[:,:,:20]
    np.save(data_dir + 'test_thingseeg2.npy', test_thingseeg2)
    np.save(data_dir + f'test_thingseeg2_avg{param}.npy', test_thingseeg2_avg)
    np.save(data_dir + f'test_thingseeg2_avg{param}_null.npy', test_thingseeg2_avg_null)

# thingseeg2 image metadata
img_metadata = np.load('data/thingseeg2_metadata/image_metadata.npy',allow_pickle=True).item()
n_train_img = len(img_metadata['train_img_concepts'])
n_test_img = len(img_metadata['test_img_concepts'])

train_img = np.zeros((n_train_img, 500, 500, 3), dtype=np.uint8)
test_img = np.zeros((n_test_img, 500, 500, 3), dtype=np.uint8)

for train_img_idx in tqdm(range(n_train_img), total=n_train_img, desc='Loading train images'):
    train_img_dir = os.path.join('data/thingseeg2_metadata', 'training_images',
        img_metadata['train_img_concepts'][train_img_idx],
        img_metadata['train_img_files'][train_img_idx])
    train_img[train_img_idx] = np.array(Image.open(train_img_dir).convert('RGB'))

np.save('data/thingseeg2_metadata/train_images.npy', train_img)

for test_img_idx in tqdm(range(n_test_img), total=n_test_img, desc='Loading test images'):
    test_img_dir = os.path.join('data/thingseeg2_metadata', 'test_images',
        img_metadata['test_img_concepts'][test_img_idx],
        img_metadata['test_img_files'][test_img_idx])
    test_img[test_img_idx] = np.array(Image.open(test_img_dir).convert('RGB'))

np.save('data/thingseeg2_metadata/test_images.npy', test_img)

test_images_dir = 'data/thingseeg2_metadata/test_images_direct/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in tqdm(range(len(test_img)), total=len(test_img), desc='Saving direct test images'):
    im = Image.fromarray(test_img[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))

# thingseeg2 concepts
n_train_img = len(img_metadata['train_img_concepts'])
n_test_img = len(img_metadata['test_img_concepts'])

train_concepts = []
test_concepts = []

for train_img_idx in tqdm(range(n_train_img), total=n_train_img, desc='Loading train images'):
    train_concepts.append(' '.join(img_metadata['train_img_concepts'][train_img_idx].split('_')[1:]))
train_concepts = np.array(train_concepts)

np.save('data/thingseeg2_metadata/train_concepts.npy', train_concepts)

for test_img_idx in tqdm(range(n_test_img), total=n_test_img, desc='Loading test images'):
    test_concepts.append(' '.join(img_metadata['test_img_concepts'][test_img_idx].split('_')[1:]))
test_concepts = np.array(test_concepts)

np.save('data/thingseeg2_metadata/test_concepts.npy', test_concepts)
