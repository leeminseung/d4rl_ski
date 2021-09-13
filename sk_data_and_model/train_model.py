import torch.nn as nn
import numpy as np
import torch
import os
import tqdm
import matplotlib.pyplot as plt
import datetime
import argparse
from utils import EarlyStopper
from utils import get_logger
from utils import get_dataloader
from SimpleFC import SimpleFC
from Simple1DCNN import Simple1DCNN
from EnsembleUsedInSKI import EntireEnsembleModel

def train_model(model, start_time, dataset, args, logger, early_stopper, device):
    loss_fn = nn.MSELoss()
    epochs = args.epochs
    train_losses = []
    val_losses = []

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(dataset, device, args)

    for epoch in tqdm.tqdm(range(epochs)):
        loss_sum = 0
        for state_and_action, next_state in train_dataloader:
            state_and_action = state_and_action.to(device)
            predicted_next_state = model(state_and_action)
            loss = loss_fn(predicted_next_state, next_state.unsqueeze(1).to(device))

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            loss_sum += loss.item()
        
        train_losses.append(loss_sum/len(train_dataloader))
        msg = '{} Epoch, Train Mean Loss: {}'.format(epoch, loss_sum/len(train_dataloader))
        logger.info(msg)

        loss_sum = 0
        with torch.no_grad():
            for state_and_action, next_state in val_dataloader:
                state_and_action = state_and_action.to(device)
                predicted_next_state = model(state_and_action)
                loss = loss_fn(predicted_next_state, next_state.unsqueeze(1).to(device))

                loss_sum += loss.item()
        
        val_losses.append(loss_sum/len(val_dataloader))
        msg = '{} Epoch, Validation Mean Loss: {}'.format(epoch, loss_sum/len(val_dataloader))
        logger.info(msg)
        
        plt.plot(train_losses, label='train loss', color='r')
        plt.plot(val_losses, label='validation loss', color='b')

        if epoch == 0:
            plt.legend()

        plt.savefig(os.path.join('model_loss', start_time, start_time + '.png'))
        np.save(os.path.join('model_loss', start_time, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join('model_loss', start_time, 'val_losses.npy'), np.array(val_losses))

        early_stopper.check_early_stopping(loss_sum/len(val_dataloader))

        if early_stopper.save_model:
            model.save_model(start_time)
            msg = '\n\n\t Best Model Saved!!! \n'
            logger.info(msg)

        if early_stopper.stop:
            msg = '\n\n\t Early Stop by Patience Exploded!!! \n'
            logger.info(msg)
            break

    return model
    
if __name__ == '__main__':
    # get Arguments
    parser = argparse.ArgumentParser(description='SKI: Traning Model Args')
    parser.add_argument('--epochs', default=300, type=int, help='Set epochs to train Model')
    parser.add_argument('--batch_size', default=3000, type=int, help='Batch size used in dataloader')
    parser.add_argument('--num_workers', default=32, type=int, help='Num workers used in dataloader')
    parser.add_argument('--input_size', default=8*6, type=int, help='Number of hidden nodes')
    parser.add_argument('--hidden_node', default=512, type=int, help='Number of hidden nodes')

    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    # make directory
    if not os.path.exists("model_loss"):
        os.mkdir("model_loss")

    if os.path.isdir(os.path.join("model_loss", start_time)) and os.path.exists(os.path.join("model_loss", start_time)):
        print('Already Existing Directory. Please wait for 1 minute.')
        exit()

    os.mkdir(os.path.join("model_loss", start_time))

    # Make and initialize model
    model = Simple1DCNN(input_size=6, hidden_size=30, output_size=1, sequential_length=4, device=device)
    # model = SimpleFC(input_size=args.input_size, hidden_size=args.hidden_node, output_size=1, device=device)
    # model = EntireEnsembleModel(input_size=args.input_size, encoder_hidden_size=args.hidden_node, encoder_output_size=args.hidden_node, transition_model_hidden_size=args.hidden_node, transition_model_output_size=1, ensemble_size=5, learning_rate=0.0005, device=device)
    model.to(device)

    # get data
    dataset = np.load("/home/lms/d4rl_ski/sk_data_and_model/step_size_1,sequential_length_5.npy")

    # record model structure
    system_logger = get_logger(name='Autoencoder model', file_path=os.path.join('model_loss', start_time, start_time + '_train_log.log'))

    # Early Stopper 
    early_stopper = EarlyStopper(patience=50)

    system_logger.info('===== Arguments information =====')
    system_logger.info(vars(args))

    system_logger.info('===== Model Structure =====')
    system_logger.info(model)

    system_logger.info('===== Loss History of Transition Model =====')
    model = train_model(model, start_time, dataset, args, system_logger, early_stopper, device)