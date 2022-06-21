import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import random
import argparse
import numpy as np
from data import *
from model import Encoder, Decoder
from sklearn.metrics import accuracy_score, f1_score

USE_CUDA = torch.cuda.is_available()


def train(config):
    train_data, word2index, tag2index, intent2index = preprocessing(config.train_data_path, config.max_length)

    if train_data == None:
        print("Please check your data or its path")
        return

    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index) // 3, config.hidden_size * 2)
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.init_weights()
    decoder.init_weights()

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)

    for step in range(config.step_size):
        losses = []
        for i, batch in enumerate(getBatch(config.batch_size, train_data)):
            x, y_1, y_2 = zip(*batch)
            x = torch.cat(x)
            tag_target = torch.cat(y_1).cuda()
            intent_target = torch.cat(y_2).cuda()
            x_mask = torch.cat([Variable(
                torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
                torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(config.batch_size, -1)
            y_1_mask = torch.cat([Variable(
                torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
                torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in tag_target]).view(config.batch_size,
                                                                                                   -1)

            encoder.zero_grad()
            decoder.zero_grad()

            output, hidden_c = encoder(x, x_mask)
            start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * config.batch_size])).cuda().transpose(1,
                                                                                                                    0) if USE_CUDA else Variable(
                torch.LongTensor([[word2index['<SOS>']] * config.batch_size])).transpose(1, 0)

            tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

            loss_1 = loss_function_1(tag_score, tag_target.view(-1))
            loss_2 = loss_function_2(intent_score, intent_target)

            loss = loss_1 + loss_2
            # losses.append(loss.data.cpu().numpy()[0] if USE_CUDA else loss.data.numpy()[0])
            losses.append(loss.cpu().detach().numpy())
            loss.backward()

            torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()

            if i % 1000 == 0:
                print("Step", step, " epoch", i, " : ", np.mean(losses))
                losses = []

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    torch.save(decoder.state_dict(), os.path.join(config.model_dir, 'jointnlu-decoder.pkl'))
    torch.save(encoder.state_dict(), os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    print("Train Complete!")


def test(config):
    test_data, word2index, tag2index, intent2index = preprocessing(config.test_data_path, config.max_length)

    if test_data is None:
        print("Please check your data or its path")
        return

    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    encoder.load_state_dict(torch.load(os.path.join(config.model_dir, 'jointnlu-encoder.pkl')))
    encoder.eval()

    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index) // 3, config.hidden_size * 2)
    decoder.load_state_dict(torch.load(os.path.join(config.model_dir, 'jointnlu-decoder.pkl')))
    decoder.eval()
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    intent_labels = []
    intent_preds = []
    slot_labels = []
    slot_preds = []
    for i, batch in enumerate(getBatch(config.batch_size, test_data)):
        x, y_1, y_2 = zip(*batch)
        x = torch.cat(x)
        tag_target = torch.cat(y_1)
        intent_target = torch.cat(y_2)
        x_mask = torch.cat([Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(config.batch_size, -1)
        y_1_mask = torch.cat([Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in tag_target]).view(config.batch_size,
                                                                                               -1)

        output, hidden_c = encoder(x, x_mask)
        start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * config.batch_size])).cuda().transpose(1, 0) \
            if USE_CUDA \
            else \
            Variable(torch.LongTensor([[word2index['<SOS>']] * config.batch_size])).transpose(1, 0)

        tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

        intent_labels.extend(list(intent_target))
        intent_preds.extend(list(np.argmax(intent_score.cpu().detach().numpy(), axis=1)))

        for label, pred in zip(tag_target.view(-1), np.argmax(tag_score.cpu().detach().numpy(), axis=1)):
            if label:
                slot_labels.append(int(label))
                slot_preds.append(pred)

        if i % 1000 == 0:
            print("epoch", i)

    print("Test Complete!")
    print("Intent Acc:", accuracy_score(intent_labels, intent_preds))
    print("Slot F1:", f1_score(slot_labels, slot_preds, average='micro'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/multiwoz/train.iob',
                        help='path of train data')
    parser.add_argument('--test_data_path', type=str, default='./data/multiwoz/test.iob',
                        help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60,
                        help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()
    train(config)
    test(config)
