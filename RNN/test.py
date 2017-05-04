
import matplotlib.pyplot as plt
from rnn_layers import *
from captioning_solver import CaptioningSolver
from rnn import CaptioningRNN
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions, fc_coco_minibatch
from image_utils import image_from_url


data = load_coco_data()

lstm_model = CaptioningRNN(
          cell_type='lstm',
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=512,
          reg=1e-8,
          dtype=np.float32,
        )

lstm_solver = CaptioningSolver(lstm_model, data,
           update_rule='rmsprop',
           num_epochs=20,
           batch_size=256,
           optim_config={
             'learning_rate': 1e-3,
           },
           verbose=True, print_every=5000,
         )

lstm_solver.train()

# Plot the training losses
plt.subplot(2, 1, 1)
plt.plot(lstm_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')

# Plot the training/validation accuracy
plt.subplot(2, 1, 2)
plt.plot(lstm_solver.train_acc_history, label='train')
plt.plot(lstm_solver.val_acc_history, label='val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()


for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(data, split=split, batch_size=20)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = lstm_model.sample(features)

    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
