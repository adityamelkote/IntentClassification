import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import set_seed, setup_gpus, check_directories
from dataloader import (
    get_dataloader,
    check_cache,
    prepare_features,
    process_data,
    prepare_inputs,
)
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel, Classifier
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR

NONE_LABELS = False
PATIENCE = 3

device = torch.device("cpu")
if torch.cuda.is_available():
    print("Cuda is available, using it")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU is available, using it")
    device = torch.device("mps")


def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], split="train")

    # task2: setup model's optimizer_scheduler if you have
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    for epoch_count in range(args.n_epochs):
        acc=0
        total=0
        losses=0
        for step, batch in enumerate(progress_bar(train_dataloader)):
            total+=1
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, split="validation")
        print("accuracy:")
        print(acc/len(datasets["train"]))
        print("epoch", epoch_count, "| losses:", losses/total)


def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()

    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.n_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)


    # Define the linear decay function
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

    scheduler = LambdaLR(optimizer, lr_lambda)


    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        acc=0
        total=0
        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            total+=1
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            
            optimizer.step()  # backprop to update the weights
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, split="validation")
        print("accuracy:")
        print(acc/len(datasets["train"]))
        print("epoch", epoch_count, "| losses:", losses/total)

def roberta_base_AdamW_grouped_LLRD(model,base_lr=1e-6,eps2=None):
        
    opt_parameters = []       # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
    
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    init_lr = base_lr/3.6
    
    for i, (name, params) in enumerate(named_parameters):  
        print(name)
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
 
        if name.startswith("encoder.embeddings") or name.startswith("encoder.encoder"):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       
            
            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr
            
            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  
            
        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("classify") or name.startswith("regressor") or name.startswith("encoder.pooler"):               
            lr = init_lr * 3.6 
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    
            
    
    return AdamW(opt_parameters, lr=init_lr,eps=eps2)
                                                                     
                                                                     
def custom_train2(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    
    optimizer = roberta_base_AdamW_grouped_LLRD(model,base_lr=args.learning_rate,eps2=args.adam_epsilon)
    
    for epoch_count in range(args.n_epochs):
        acc=0
        total=0
        losses=0
        for step, batch in enumerate(progress_bar(train_dataloader)):
            total+=1
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, split="validation")
        print("accuracy:")
        print(acc/len(datasets["train"]))
        print("epoch", epoch_count, "| losses:", losses/total)

def custom_train3(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()

    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler
    optimizer = roberta_base_AdamW_grouped_LLRD(model,base_lr=args.learning_rate,eps2=args.adam_epsilon)
    #optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.n_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Define the linear decay function
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

    scheduler = LambdaLR(optimizer, lr_lambda)
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        acc=0
        model.train()
        total=0
        for step, batch in enumerate(progress_bar(train_dataloader)):
            total+=1
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            
            optimizer.step()  # backprop to update the weights
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        # Evaluate the model on the validation set after each epoch
        run_eval(args, model, datasets, tokenizer, split='validation')
        print("accuracy:")
        print(acc/len(datasets["train"]))
        print('epoch', epoch_count, '| losses:', losses/total)
        

def run_eval_con(
    model, args, classifier_model, datasets, tokenizer, split="validation"
):
    print("run eval con")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    classifier_model.eval()

    dataloader = get_dataloader(args, datasets[split], split)
    avg_loss = 0
    acc = 0
    total = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        if NONE_LABELS:
            encodings = model(inputs, None, inference=True)
        else:
            encodings = model(inputs, labels, inference=True)
        logits = classifier_model(encodings)
        loss = criterion(logits, labels)

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
        avg_loss += loss.item()
        total = total + 1
    avg_loss = avg_loss / total
    print("")
    print(
        f"{split} acc:",
        acc / len(datasets[split]),
        f"|dataset split {split} size:",
        len(datasets[split]),
        f" | avg loss: {avg_loss}",
    )
    return avg_loss


   
def run_eval(args, model, datasets, tokenizer, split="validation"):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    avg_loss = 0
    acc = 0
    total = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = None
        if NONE_LABELS:
            logits = model(inputs, None)
        else:
            logits = model(inputs, labels)
        loss = criterion(logits, labels)

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
        avg_loss += loss.item()
        total = total + 1
    avg_loss = avg_loss / total
    print("")
    print(
        f"{split} acc:",
        acc / len(datasets[split]),
        f"|dataset split {split} size:",
        len(datasets[split]),
        f" | avg loss: {avg_loss}",
    )


def roberta_base_AdamW_grouped_LLRD(model, base_lr=1e-6, eps2=None):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    init_lr = base_lr / 3.6

    for i, (name, params) in enumerate(named_parameters):
        #         print(name)
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if name.startswith("encoder.embeddings") or name.startswith("encoder.encoder"):
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = 0

            # For set_2, increase lr to 0.00000175
            # lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

            # For set_3, increase lr to 0.0000035
            # lr = init_lr * 3.5 if any(p in name for p in set_3) else lr

            opt_parameters.append(
                {"params": params, "weight_decay": weight_decay, "lr": lr}
            )

        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).
        if (
            name.startswith("classify")
            or name.startswith("regressor")
            or name.startswith("encoder.pooler")
        ):
            lr = init_lr * 3.6

            opt_parameters.append(
                {"params": params, "weight_decay": weight_decay, "lr": lr}
            )

    return AdamW(opt_parameters, lr=init_lr, eps=eps2)


def supcon_train(classifier_model, args, model, datasets, tokenizer):
    from loss import SupConLoss

    # TODO pass labels if doing supcon none if simCLR?
    criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset
    print("training supcon model temp=", args.temperature)

    train_dataloader = get_dataloader(args, datasets["train"], split="train")

    # task2: setup optimizer_scheduler in your model
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # todo? add scheduler? unsure

    # task3: write a training loop for SupConLoss function
    model.train()

    #      TODO Put this in EPOCHS
    for epoch_count in range(args.n_epochs):
        losses = []
        model.train()
        for idx, batch in enumerate(progress_bar(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)

            if NONE_LABELS:
                f1 = model(inputs, None)
                f2 = model(inputs, None)
            else:
                f1 = model(inputs, labels)
                f2 = model(inputs, labels)

            """ 
             features is the embedding for the input,
             we will have batch_size * feature_dim in features
            """
            f1 = f1.unsqueeze(1)
            f2 = f2.unsqueeze(1)

            features = torch.cat((f1, f2), dim=1)

            if NONE_LABELS:
                loss = criterion(features, None)
            else:
                loss = criterion(features, labels)

            # update metric
            losses.append(loss.item())

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #         run_eval(args, model, datasets, tokenizer, split="validation")
        print("epoch", epoch_count, "| loss", np.mean(losses))

    print("switching to training classifier")

    for param in model.parameters():
        param.requires_grad = False

    train_dataloader = get_dataloader(args, datasets["train"], split="train")
    optimizer = AdamW(
        classifier_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon
    )
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()

    prev_loss = 10000

    model.eval()
    for epoch_count in range(args.n_epochs):
        losses = 0
        classifier_model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            encodings = model(inputs, labels, inference=True)

            logits = classifier_model(encodings)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        next_loss = run_eval_con(
            model, args, classifier_model, datasets, tokenizer, split="validation"
        )
        run_eval_con(model, args, classifier_model, datasets, tokenizer, split="train")
        if next_loss > prev_loss:
            pat += 1
        else:
            pat = 0
        prev_loss = next_loss
        if pat >= PATIENCE:
            print("early exit")
            break
        print("epoch", epoch_count, "| losses:", losses)


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    print(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()

        (data)
        features = prepare_features(args, data, tokenizer, cache_results)

    datasets = process_data(args, features, tokenizer)

    for k, v in datasets.items():
        print(k, len(v))

    if args.task == "baseline":
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split="test")
    elif (
        args.task == "custom"
    ):  # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        custom_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split="test")
    elif (
        args.task == "custom2"
    ):  # you can have multiple custom task for different techniques
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        custom_train2(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split="test")
        
    elif (
        args.task == "custom3"
    ):  # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        custom_train3(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split="test")
        
    elif args.task == "supcon":
        classifier_model = Classifier(args, target_size=60).to(device)
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        #         run_eval(args, classifier_model, datasets, tokenizer, split="validation")
        run_eval_con(model, args, classifier_model, datasets, tokenizer, split="test")
        supcon_train(classifier_model, args, model, datasets, tokenizer)
        run_eval_con(model, args, classifier_model, datasets, tokenizer, split="test")

