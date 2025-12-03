"""Repeatable code parts concerning optimization and training schedules."""

import torch
import numpy as np
from collections import defaultdict

from .utils import print_and_save_stats
from ..consts import NON_BLOCKING, BENCHMARK, PIN_MEMORY
from ..sponge.energy_estimator import check_sourceset_consumption
import torch.optim as optim
# Pretrained model
from torchvision import models
import torch.nn as nn
import os
import sys
from os.path import abspath
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import warnings
warnings.filterwarnings('ignore')
from art import config
from art.utils import load_dataset, get_file
from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import FeatureCollisionAttack
from ..hyperparameters import training_strategy
from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from art.attacks.poisoning import GradientMatchingAttack

from forest.utils import *
torch.backends.cudnn.benchmark = BENCHMARK
from art.utils import to_categorical
from PIL import Image
from numpy import asarray
from skimage.transform import resize

def add_trigger_torch(images, trigger_position=(0, 0), trigger_size=3, trigger_value=1.0):
    # Function to evaluate the model on a triggered input
    
    """
    Adds a square trigger to images using PyTorch Tensors.
    :param images: torch.Tensor of shape (num_images, channels, height, width)
    :param trigger_position: (x, y) tuple for the top-left position of the trigger
    :param trigger_size: The size of the square trigger
    :param trigger_value: The pixel value for the trigger (e.g., 1.0 for white if normalized)
    :return: images with trigger added
    """
    images_with_trigger = images.clone()  # Copy to avoid modifying original images
    num_images, channels, height, width = images.shape
    for img in images_with_trigger:
        # Apply the trigger as a white square in the specified position
        img[:, trigger_position[0]:trigger_position[0] + trigger_size, 
            trigger_position[1]:trigger_position[1] + trigger_size] = trigger_value
    return images_with_trigger


def get_input_output(kettle, model, optimizer, pretraining_phase=False):
    # poisoned_model = copy.deepcopy(model)
    # The same
    if pretraining_phase:
        train_loader = kettle.pretrainloader
        valid_loader = kettle.validloader
    else:
        if kettle.args.ablation < 1.0:
            # run ablation on a subset of the training set
            train_loader = kettle.partialloader
        else:
            train_loader = kettle.trainloader
        valid_loader = kettle.validloader
    # print(f"train_loader shape: {train_loader.shape}")
    # print(f"valid_loader shape: {valid_loader.shape}")
    input_batches = []
    label_batches = []
    # the same
    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(valid_loader):
            # process data
            test_inputs = inputs.to(**kettle.setup)
            test_labels = labels.to(device=kettle.setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)  
            input_batches.append(test_inputs)
            label_batches.append(test_labels)
            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1} batches')
        x_test = torch.cat(input_batches, dim=0)
        y_test = torch.cat(label_batches, dim=0)
        
        print(f"All x_test shape: {x_test.shape}")
        print(f"All y_test shape: {y_test.shape}")
       
        # Since inputs are images (tensors of shape [512, 3, 32, 32]), you may want to inspect individual images or just their pixel values
        # To check the min and max pixel values in inputs (across all images)
        inputs_min = test_inputs.min().item()  # Get minimum pixel value
        inputs_max = test_inputs.max().item()  # Get maximum pixel value
        print(f"Min pixel value in inputs_min: {inputs_min}, Max pixel value in inputs_max: {inputs_max}")
    train_input_batches = []
    train_label_batches = []
    train_batch_count = 0

    for batch, (inputs, labels, ids) in enumerate(train_loader):
        train_batch_count += 1

        # Prep Mini-Batch
        # Exra
        # to_sponge = [i for i, idx in enumerate(ids.tolist()) if idx in poison_ids]
        # the same
        optimizer.zero_grad()
        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
        train_input_batches.append(inputs)
        train_label_batches.append(labels)
        # Added from other codes start
        mydefs = training_strategy(kettle.args.net[0], kettle.args)    
        optimizer, scheduler = get_optimizers(model, kettle.args, mydefs)  
        if (batch + 1) % 100 == 0:
            print(f'Processed {i + 1} batches')
    print(f"Total training batches processed: {train_batch_count}")

    x_train = torch.cat(train_input_batches, dim=0)
    y_train = torch.cat(train_label_batches, dim=0)   
    print(f"All x_train shape: {x_train.shape}")
    print(f"All y_train shape: {y_train.shape}")        
    print(f"inputs shape: {inputs.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"Total samples in train_loader: {len(train_loader.dataset)}")
    print(f"Total samples in valid_loader: {len(valid_loader.dataset)}")
    print(f"Batch size in train_loader: {train_loader.batch_size}")
    print(f"Batch size in valid_loader: {valid_loader.batch_size}")

    # Calculate mean and standard deviation across all the images
    mean = torch.mean(x_train)
    std = torch.std(x_train)
    # for data, target in train_loader:
    #     print(target)  # Add this line to print the target labels
    print(f'Mean: {mean.item()}, Std: {std.item()}')
    return train_loader, valid_loader, x_train, y_train, x_test, y_test, optimizer, scheduler


def poisoning_attack(attack_type, kettle, classifier, x_train_poisoned, y_train_poisoned, x_test, y_test):
    # Variables
    num_classes = kettle.args.sigma  # Number of target and base classes
    percentage =  kettle.args.budget  # Percentage of instances to poison per class combination
    np.random.seed(301)
    N_CLASSES = 200
    class_descr = [str(i) for i in range(N_CLASSES)]

    # class_descr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    unique_classes = np.unique(np.argmax(y_train_poisoned, axis=1))
    assert num_classes * 2 <= len(unique_classes), "Not enough unique classes available."
    # Randomly select target and base classes
    np.random.shuffle(unique_classes)
    unique_classes = np.unique(np.argmax(y_train_poisoned, axis=1))
    N_CLASSES = len(unique_classes)   # for Tiny -> 200
    num_target = min(5, N_CLASSES // 2)
    num_base   = num_target
    np.random.shuffle(unique_classes)

    target_classes = unique_classes[:num_target]
    base_classes   = unique_classes[num_target:num_target + num_base]


    # target_classes = unique_classes[:num_classes]
    # base_classes = unique_classes[num_classes:num_classes*2]
    # Function to determine the minimum available instances among selected classes
    def minimum_instances(classes):
        min_size = np.inf
        for cls in classes:
            class_size = len(np.where(np.argmax(y_train_poisoned, axis=1) == cls)[0])
            if class_size < min_size:
                min_size = class_size
        return min_size
    # Calculate the portion based on the smallest class size
    min_size_target = minimum_instances(target_classes)
    min_size_base = minimum_instances(base_classes)
    min_size = min(min_size_target, min_size_base)
    portion = int(np.floor(min_size * (percentage / 100))) 
    # portion = 2
    # Function to select indices for classes
    def select_indices_for_classes(classes, portion, y_data):
        selected_indices = {}
        for cls in classes:
            class_indices = np.where(np.argmax(y_data, axis=1) == cls)[0]
            if len(class_indices) < portion:
                print(f"Not enough data for class {cls}, using available {len(class_indices)} instances.")
                selected_indices[cls] = class_indices
            else:
                selected_indices[cls] = np.random.choice(class_indices, portion, replace=False)
        return selected_indices
    # Calculate portions for each class and select indices
    target_indices = select_indices_for_classes(target_classes, portion, y_train_poisoned)
    base_indices = select_indices_for_classes(base_classes, portion, y_train_poisoned)
    total_portion = len(base_indices)
    print(f"Using {portion} instances from each class based on {percentage}% of the smallest class size and total of {total_portion}.")
    feature_layer = classifier.layer_names[-2]
    # Perform poisoning attacks between each target and base class
    # Initialize storage for poisoned data
    total_poisoned_samples = len(target_classes) * len(base_classes) * portion
    poisoned_data = np.zeros((total_poisoned_samples, *x_train_poisoned.shape[1:]), dtype=x_train_poisoned.dtype)
    poisoned_labels = np.zeros((total_poisoned_samples, y_train_poisoned.shape[1]), dtype=y_train_poisoned.dtype)
    

    # Track the current index in the storage arrays
    current_index = 0
    # poisoned_data, poisoned_labels = [], []
    for target_cls in target_classes:
        for base_cls in base_classes:
            target_idx = target_indices[target_cls]
            base_idx = base_indices[base_cls]    
            for i in range(portion):
                print (f"rnning for instance:{i}, target_cls: {target_cls}, base_cls:{base_cls}")
                target_instance = np.expand_dims(x_train_poisoned[target_idx[i]], axis=0)
                base_instance = np.expand_dims(x_train_poisoned[base_idx[i]], axis=0)
                # print(f"target_cls: {target_cls}")

                y_trigger_instance = to_categorical(np.array([target_cls]), 200)
                # print((base_instance.shape), (target_instance.shape))

                if attack_type == "FeatureCollision":
                    # print('attack_type = "FeatureCollision"')
                    attack = FeatureCollisionAttack(
                        classifier, 
                        target_instance, 
                        feature_layer, 
                        max_iter=10, 
                        similarity_coeff=256,
                        watermark=0.3,
                        learning_rate=1)
                    poison_instance, _ = attack.poison(base_instance)
                
                elif attack_type == "MatchingGradient":
                    print("Matching Gradient Attak starts!")
                    eps = 0.01 / 0.50

                    # epsilson = 0.01 / (0.2822 if kettle.args.dataset == "CIFAR10" else 0.8932 + 1e-7)
                    attack = GradientMatchingAttack(classifier=classifier,
                        percent_poison=0.5,
                        max_trials=8,
                        max_epochs=20,
                        clip_values=(0, 1),
                        epsilon=epsilson,
                        verbose=False)

                    all_poisoned_instances, all_poisoned_labels = attack.poison(base_instance, y_trigger_instance, x_train_poisoned, y_train_poisoned)
                    # Assuming you want only the poisoned version of `base_instance`, select it here
                    poison_instance = all_poisoned_instances[base_idx[i]:base_idx[i] + 1]  # Shape should be (1, 3, 32, 32)
                    poison_labels = all_poisoned_labels[base_idx[i]:base_idx[i] + 1]  # Shape should be (1, 10) if one-hot encoded
                    # print((poison_instance.shape), (poison_labels.shape))
                    # For Matching Gradient Attack
                    if isinstance(poison_instance, np.ndarray) or isinstance(poison_instance, torch.Tensor):
                        print(f"poison_instance shape (Matching Gradient Attack): {poison_instance.shape}")
                    
                    if isinstance(poison_labels, np.ndarray) or isinstance(poison_labels, torch.Tensor):
                        print(f"poison_labels shape (Matching Gradient Attack): {poison_labels.shape}")

                    # poisoned_data.append(poison_instance[0])  # Appends the single instance without batch dimension
                    # poisoned_labels.append(poison_labels[0])
                    # print((poisoned_data.shape), (poisoned_labels.shape))

                else:
                    x_trigger  = np.copy(base_instance)
                    # print('attack_type = "SleeperAgent"')
                    y_trigger  = to_categorical([target_cls], nb_classes=200)
                    y_trigger = np.tile(y_trigger,(len(base_indices),1))
                    if y_trigger.ndim == 2:  # If the labels are one-hot encoded
                        y_trigger = torch.tensor(y_trigger)  # Convert NumPy array to PyTorch Tensor
                        y_trigger = torch.argmax(y_trigger, dim=1)  # Now you can apply torch.argmax

                    print(f"x_trigger shape: {x_trigger.shape}")
                    print(f"y_trigger shape: {y_trigger.shape}")

                   
                    index_target = np.copy(target_idx)
                    class_source = base_cls
                    class_target = target_cls
                    patch_size = 8
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    img = Image.open(r'C:\Users\narge\sponge\data\trigger_10.png')
                    numpydata = asarray(img)
                    patch = resize(numpydata, (patch_size,patch_size,3))
                    patch = np.transpose(patch,(2,0,1))



                    
                    attack = SleeperAgentAttack(classifier,
                                percent_poison=0.50,
                                max_trials=1,
                                max_epochs=100,
                                learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), [250, 350, 400, 430, 460]),
                                epsilon=16/255,
                                batch_size=8,
                                verbose=1,
                                indices_target=index_target,
                                patching_strategy="random",
                                selection_strategy="max-norm",
                                patch=patch,
                                retraining_factor = 4,
                                model_retrain = True,
                                model_retraining_epoch = 80,
                                retrain_batch_size = 128,
                                class_source = class_source,
                                class_target = class_target,
                                device_name = str(device)        
                           )
                    poison_instance, poison_labels = attack.poison(x_trigger,y_trigger,x_train_poisoned,y_train_poisoned,x_test,y_test)                 
                # Store poisoned data
                
                # for match gradient attack
                # Assuming pre-allocated arrays poisoned_data and poisoned_labels
                # poisoned_data[current_index] = poison_instance[0]  # Removing the extra batch dimension
                # poisoned_labels[current_index] = poison_labels[0]
                # # Print the shapes of the current poisoned instance and label after assignment
                # print(f"poisoned_data[{current_index}] shape: {poisoned_data[current_index].shape}")
                # print(f"poisoned_labels[{current_index}] shape: {poisoned_labels[current_index].shape}")

                # current_index += 1

                # for other
                poisoned_data[current_index] = poison_instance
                poisoned_labels[current_index] = y_train_poisoned[base_idx[i]]
                current_index += 1   
    # Replace the original data in the dataset
    for idx, (target_cls, base_cls) in enumerate(zip(target_classes, base_classes)):
        start_idx = idx * portion
        end_idx = start_idx + portion
    
        print(f"poisoned_data[start_idx:end_idx] shape: {poisoned_data[start_idx:end_idx].shape}")
        print(f"poisoned_labels[start_idx:end_idx] shape: {poisoned_labels[start_idx:end_idx].shape}")
    
        x_train_poisoned[base_indices[base_cls]] = poisoned_data[start_idx:end_idx]
        y_train_poisoned[base_indices[base_cls]] = poisoned_labels[start_idx:end_idx]
    
    # Convert x_test to tensor and add trigger
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    print(f"x_test_tensor shape before adding trigger: {x_test_tensor.shape}")
    target_label = 1  
    x_test_poisoned = add_trigger_torch(x_test_tensor)
    y_test_poisoned = torch.full((len(x_test_poisoned),), target_label, dtype=torch.long)
    
    print(f"y_test_poisoned shape: {y_test_poisoned.shape}")
    print(f"x_train_poisoned final shape: {x_train_poisoned.shape}")
    print(f"y_train_poisoned final shape: {y_train_poisoned.shape}")
    
    print(f"Poisoning {attack_type} completed for all class pairs.")
    
    return x_train_poisoned, y_train_poisoned, x_test_poisoned, y_test_poisoned, class_descr

def final_attack(attack_type, kettle, model, x_train, y_train, x_test,y_test, optimizer):
    num_classes = 200
    x_train = x_train.cpu().numpy()  # Convert to NumPy array if needed
        # outputs = outputs.cpu().numpy()  # Convert to NumPy array if needed
    x_test = x_test.cpu().numpy()
    y_train = y_train.cpu().numpy()  # Convert to NumPy array if needed
    y_train = np.eye(num_classes)[y_train]
    
    y_test = y_test.cpu().numpy()  # Convert to NumPy array if needed
    y_test = np.eye(num_classes)[y_test]
    
    # Attack should be added here
    # clean_classifier = PyTorchClassifier(
    #     model=model,
    #     clip_values=(0, 1),
    #     preprocessing=((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    #     nb_classes=10,
    #     input_shape=(3, 32, 32),
    #     loss=nn.CrossEntropyLoss(),
    #     optimizer=torch.optim.Adam(model.parameters(), lr=0.0001))
    print('attacker type is:', attack_type)
    if attack_type == "SleeperAgent": 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))
        # Convert NumPy float to PyTorch tensor and move to device
        mean_tensor = torch.tensor(mean, dtype=torch.float32).to(device)
        std_tensor = torch.tensor(std, dtype=torch.float32).to(device)
        mean_np = mean_tensor.cpu().numpy()  # Move to CPU first, then convert to NumPy
        std_np = std_tensor.cpu().numpy()    # Move to CPU first, then convert to NumPy


        poisoned_classifier = PyTorchClassifier(model, input_shape=(3, 32, 32), loss=nn.CrossEntropyLoss(), 
                                       optimizer=torch.optim.Adam(model.parameters(), lr=0.0001), nb_classes=200, clip_values=(0, 1), 
                                       preprocessing=(mean_np, std_np))
        
        for name, param in poisoned_classifier.model.named_parameters():
            if not param.requires_grad:
                print(f"Parameter {name} is frozen and will not receive gradients.")
            if "layer_to_unfreeze" in name:
                param.requires_grad = True
        for param in poisoned_classifier.model.parameters():
            param.requires_grad = True

        # Convert NumPy arrays to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Assuming `model` is your neural network, define optimizer and criterion
        optimizer = torch.optim.Adam(poisoned_classifier.model.parameters(), lr=0.001)  # Use the correct model's parameters
        criterion = nn.CrossEntropyLoss()  # Assuming classification problem
       
        # poisoned_classifier.model = poisoned_classifier.model.to(device)


        # Set the model to training mode
        poisoned_classifier.model.train()
        
        # Training loop with gradient handling
        num_epochs = 10
        batch_size = 8
        num_batches = len(x_train_tensor) // batch_size
        
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(x_train_tensor))
                inputs = x_train_tensor[start_idx:end_idx]
                inputs = inputs.to(device)

                labels = y_train_tensor[start_idx:end_idx]
                labels = labels.to(device)

        
                # Zero the gradients
                optimizer.zero_grad()
        
                # Forward pass
                outputs = poisoned_classifier.model(inputs.to(device))  # Make sure `inputs` is moved to the device

        
                # Ensure labels are in the correct form (class indices)
                if labels.dim() == 2 and labels.size(1) > 1:  # Check if labels are one-hot encoded
                    labels = torch.argmax(labels, dim=1)
                
                # Convert labels to long type if they are not already
                labels = labels.long()
                
                # Now compute the loss
                loss = criterion(outputs, labels)
                                # Backward pass (computes gradients)
                loss.backward()
                for name, param in poisoned_classifier.model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradient for {name} is computed.")
                    else:
                        print(f"Warning: Gradient for {name} is None.")

        
                # Update the model parameters
                optimizer.step()
        
            # Optionally, print loss or accuracy after each epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        
        # Set the model to evaluation mode
        poisoned_classifier.model.eval()
        
       # Predictions on test data (in batches)
        predicted_classes = []
        with torch.no_grad():  # Disable gradient calculation for inference
            num_batches = len(x_test_tensor) // batch_size
            for batch_idx in range(num_batches + 1):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(x_test_tensor))
                inputs = x_test_tensor[start_idx:end_idx]
        
                outputs = poisoned_classifier.model(inputs.to(device))
                predicted_classes_batch = torch.argmax(outputs, dim=1).cpu().numpy()
                predicted_classes.append(predicted_classes_batch)
        
        # Flatten predicted classes
        predicted_classes = np.concatenate(predicted_classes)
        
        # Calculate accuracy
        accuracy = np.sum(predicted_classes == y_test_tensor.cpu().numpy()) / len(y_test_tensor)
        print(f"Accuracy on benign test examples: {accuracy * 100:.2f}%")

    else:
        TINY_MEAN = (0.480, 0.448, 0.397)
        TINY_STD  = (0.277, 0.269, 0.282)

        poisoned_classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            preprocessing=(TINY_MEAN, TINY_STD),
            nb_classes=200,
            input_shape=(3, 64, 64),
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        )

        # poisoned_classifier = PyTorchClassifier(
        #     model=model,
        #     clip_values=(0, 1),
        #     preprocessing=((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        #     nb_classes=10,
        #     input_shape=(3, 32, 32),
        #     loss=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
        # )
        # poisoned_classifier = PyTorchClassifier(
        #     model=model,
        #     clip_values=(0, 1),  # MNIST pixel values are between 0 and 1
        #     preprocessing=((0.1000,), (0.2752,)),  # Mean and std for MNIST grayscale images
        #     nb_classes=10,  # MNIST has 10 classes (digits 0-9)
        #     input_shape=(1, 28, 28),  # MNIST images are 28x28 with 1 channel (grayscale)
        #     loss=nn.CrossEntropyLoss(),  # Cross-entropy loss for classification
        #     optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer
        # )

        
    try:
        num_ftrs = model.dnn.classifier[0][0].in_features
    except AttributeError as e:
        print(f"Error accessing the classifier layer's in_features: {str(e)}")
        num_ftrs = None

    model.fc = nn.Linear(num_ftrs, 200)  # Adjust output to 10 classes for CIFAR-10, mnist
    
    def evaluate_model(classifier, x_test, y_test):
        
         # Predictions on test data (in batches)
         predicted_classes = []
         batch_size = 8
         x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
         y_test_tensor = torch.tensor(y_test, dtype=torch.long)
         num_batches = len(x_test_tensor) // batch_size
     
         with torch.no_grad():  # Disable gradient calculation for inference
             for batch_idx in range(num_batches + 1):
                 start_idx = batch_idx * batch_size
                 end_idx = min(start_idx + batch_size, len(x_test_tensor))
                 inputs = x_test_tensor[start_idx:end_idx]
                 outputs = classifier.model(inputs.to('cuda'))
                 predicted_classes_batch = torch.argmax(outputs, dim=1).cpu().numpy()
                 predicted_classes.append(predicted_classes_batch)
     
         predicted_classes = np.concatenate(predicted_classes)
         accuracy = np.sum(predicted_classes == y_test_tensor.cpu().numpy()) / len(y_test_tensor)
         
    
         print(f"Accuracy: {accuracy * 100:.2f}%")
         return accuracy
    
     # Call the evaluate_model function with the classifier and datasets
    x_train_poisoned, y_train_poisoned, x_test_poisoned, y_test_poisoned, class_descr = poisoning_attack(attack_type, kettle, poisoned_classifier, x_train, y_train, x_test, y_test)

    # Check if y_train_poisoned is one-hot encoded and convert if true
    if y_train_poisoned.ndim == 2 and y_train_poisoned.shape[1] == num_classes:
        y_train_indices = np.argmax(y_train_poisoned, axis=1)
    else:
        y_train_indices = y_train_poisoned  # Assuming it is already in class index form
    
    # Check if y_test_poisoned is one-hot encoded and convert if true
    if y_test_poisoned.ndim == 2 and y_test_poisoned.shape[1] == num_classes:
        y_test_indices = np.argmax(y_test_poisoned, axis=1)
    else:
        y_test_indices = y_test_poisoned  # Assuming it is already in class index form
    
    # Train the classifier on clean data
    poisoned_classifier.fit(x_train, y_train, batch_size=512, nb_epochs=100)
    
    # Evaluate on clean data before the attack
    print("Evaluating model on clean data before attack...")
    accuracy_before = evaluate_model(poisoned_classifier, x_test, y_test)
    print(f"Accuracy before attack: {accuracy_before * 100:.2f}%")
    
    # Convert indices to tensors and move to device
    y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long).to('cuda')
    x_train_tensor = torch.tensor(x_train_poisoned, dtype=torch.float32).to('cuda')
    
    y_test_tensor = torch.tensor(y_test_indices, dtype=torch.long).to('cuda')
    x_test_tensor = torch.tensor(x_test_poisoned, dtype=torch.float32).to('cuda')

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, class_descr

    
def run_poisoning_step(kettle, train_loader, valid_loader, x_train_poisoned, y_train_poisoned, class_descr, test_inputs, test_labels, opt, schedulerr, 
                       poison_delta, epoch, stats, model, defs, optimizer, scheduler, loss_fn , pretraining_phase=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    x_train_poisoned = x_train_poisoned.to(device)
    y_train_poisoned = y_train_poisoned.to(device)
    epoch_loss, total_preds, correct_preds = 0, 0, 0
    
    # Extra
    # poison_delta, poison_ids = delta
    victim_leaf_nodes = [module for module in model.modules()
                         if len(list(module.children())) == 0] 
    
    stats['Poison_loss'].append(0)
    stats['epoch_loss'].append(0)
    # Extra

    # print("Evaluating clean model:")
    # clean_predictions, clean_accuracy, clean_loss = evaluate_model(clean_classifier, x_test_poisoned, y_test_poisoned, class_descr)
    
    # poisoned_classifier.fit(x_train_poisoned, y_train_poisoned, nb_epochs=20, batch_size=256)
    import math
    batch_size = 8

    num_samples = x_train_poisoned.shape[0]
    num_batches = math.ceil(num_samples / batch_size)  # Use ceiling to include all samples

    for i in range(num_batches):
        # Prep Mini-Batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)  # Avoid going out of bounds
        # Extract the batch from the dataset
        # print(f"Starting batch {i+1}/{num_batches}")



        inputs = x_train_poisoned[start_idx:end_idx]
        labels = y_train_poisoned[start_idx:end_idx]
        # print(f"Processing batch {i+1} with {inputs.shape[0]} samples.")


        optimizer.zero_grad()
        model.module.dnn.classifier.train() if model.frozen else model.train()
    
        def criterion(outputs, labels):
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs.data, dim=1)
            correct_preds = (predictions == labels).sum().item()
            return loss, correct_preds

        
        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(inputs)

        loss, preds = criterion(outputs, labels)
        correct_preds += preds
        # Zero the gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        
        

        # Optionally print statistics
        # if i % 100 == 0:
        #     print(f'Epoch {epoch + 1}, Batch {i + 1}/{num_batches}, Loss: {loss.item()}')

        total_preds += labels.shape[0]
        #     # After training with poisoned data
        # print("Evaluating poisoned model:")
        # predictions, accuracy, loss = evaluate_model(poisoned_classifier, x_test_poisoned, y_test_poisoned, class_descr)
        # Added for attack training end
    
        # Exra - calculating the energy layers -should be after attack generated
        stats['Poison_loss'][-1] += loss.item()
    
        if isinstance(poison_delta, str) and poison_delta == 'Poisoning':
            # clean_loss, patched_loss, backdoor_loss, step_stats = sponge_step_loss(model, inputs[to_sponge],
            #                                                                       labels[to_sponge], victim_leaf_nodes,
            #                                                                       loss_fn)
            
            # instead of regular model we should add the attacked model I GUESS
            sponge_loss, sponge_stats = sponge_step_loss(model, x_train_poisoned, victim_leaf_nodes, kettle.args)
            stats['Poison'].append(sponge_stats)
            stats['Poison_loss'][-1] += sponge_stats['sponge_stats'][0]
            loss = loss - sponge_loss
            
        #the same
    
        loss.backward()
        epoch_loss += loss.item()
    
        optimizer.step()
    
        if defs.scheduler == 'cyclic':
            scheduler.step()
        # print(f"Ending batch {i+1}/{num_batches}")
    # if kettle.args.dryrun:
    #     break
    # print(f"inputs shape: {inputs.shape}")
    # print(f"labels shape: {labels.shape}")
    # Extra
    stats['Poison_loss'][-1] /= len(train_loader)
    stats['epoch_loss'][-1] /= len(train_loader)
    
    # the same
    if defs.scheduler == 'linear':
        scheduler.step()
    if defs.scheduler == 'exponential':
        scheduler.step()
    
    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        poisoned_model, predictions, valid_loss = run_poison_validation(model, loss_fn, valid_loader, kettle.setup, kettle.args.dryrun)
        _, energy_ratio = check_sourceset_consumption(model, kettle, stats)
    else:
        predictions, valid_loss = None, None
        energy_ratio = 0
        
    
    # source_acc, source_loss accuracy and loss for validation source samples with backdoor trigger
    current_lr = optimizer.param_groups[0]['lr']
    # print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
    #                      predictions, valid_loss, energy_ratio)
            


def sponge_step_loss(model, inputs, victim_leaf_nodes, args):
    sponge_loss, _, sponge_stats = data_sponge_loss(model, inputs, victim_leaf_nodes, args)
    sponge_stats = dict(sponge_loss=float(sponge_loss), sponge_stats=sponge_stats)
    return sponge_loss, sponge_stats

def data_sponge_loss(model, x, victim_leaf_nodes, args):
    sponge_stats = SpongeMeter(args)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)
    if isinstance(x, np.ndarray):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.tensor(x, dtype=torch.float32, device=device)  # Replace 'your_device' with 'cuda' or 'cpu' as appropriate


    outputs = model(x)

    sponge_loss = fired_perc = fired = l2 = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i].to('cuda')
        fired += float(sponge_stats.fired[i])
        fired_perc += float(sponge_stats.fired_perc[i])
        l2 += float(sponge_stats.l2[i])
    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)
    fired_perc /= len(sponge_stats.loss)

    sponge_loss *= args.lb
    return sponge_loss, outputs, (float(sponge_loss), fired, fired_perc, l2)    
    
        
    
def run_step(kettle, poison_delta, epoch, stats, model, defs, optimizer, scheduler, loss_fn, pretraining_phase=False):
    epoch_loss, total_preds, correct_preds = 0, 0, 0

    if pretraining_phase:
        train_loader = kettle.pretrainloader
        valid_loader = kettle.validloader
        
    else:
        if kettle.args.ablation < 1.0:
            # run ablation on a subset of the training set
            train_loader = kettle.partialloader
        else:
            train_loader = kettle.trainloader
        valid_loader = kettle.validloader

    for batch, (inputs, labels, ids) in enumerate(train_loader):
        # Prep Mini-Batch
        optimizer.zero_grad()
        

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
        # infer the number of classes from the model output
        with torch.no_grad():
            num_classes = model(inputs[:1]).size(1)
        
        if labels.max() >= num_classes or labels.min() < 0:
            print(f"[Label ERROR] Found labels out of range 0–{num_classes-1}:",
                  labels[(labels >= num_classes) | (labels < 0)])

        # #### Add poison pattern to data #### #
        if poison_delta is not None:
            poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            if len(batch_positions) > 0:
                inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)
    
        # Switch into training mode
        # list(model.children())[-1].train() if model.frozen else model.train()

        model.module.dnn.classifier.train() if model.frozen else model.train()
    # for data, target in train_loader:
    #     print(target)
    #     # Ensure that target values are between 0 and 42

        def criterion(outputs, labels):
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs.data, dim=1)
            correct_preds = (predictions == labels).sum().item()
            return loss, correct_preds

        # Do normal model updates, possibly on modified inputs
        outputs = model(inputs)
        loss, preds = criterion(outputs, labels)
        correct_preds += preds

        total_preds += labels.shape[0]

        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()
    if defs.scheduler == 'exponential':
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        predictions, valid_loss = run_validation(model, loss_fn, valid_loader, kettle.setup, kettle.args.dryrun)
        _, energy_ratio = check_sourceset_consumption(model, kettle, stats)
    else:
        predictions, valid_loss = None, None
        energy_ratio = 0

    # source_acc, source_loss accuracy and loss for validation source samples with backdoor trigger
    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         predictions, valid_loss, energy_ratio)

def default_value():
    return {'correct': 0, 'total': 0}

def run_poison_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader.

    Hint: The validation numbers in "target" and "source" explicitely reference the first label in target_class and
    the first label in source_class."""
    model.eval()
    predictions = defaultdict(default_value)
    loss = 0
    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            predictions['all']['total'] += labels.shape[0]
            predictions['all']['correct'] += (predicted == labels).sum().item()

            if dryrun:
                break

    for key in predictions.keys():
        if predictions[key]['total'] > 0:
            predictions[key]['avg'] = predictions[key]['correct'] / predictions[key]['total']
        else:
            predictions[key]['avg'] = float('nan')

    loss_avg = loss / (i + 1)
    return model, predictions, loss_avg

def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader.

    Hint: The validation numbers in "target" and "source" explicitely reference the first label in target_class and
    the first label in source_class."""
    model.eval()
    # predictions = defaultdict(lambda: dict(correct=0, total=0))
    # Define a function that returns the default dictionary
    
    
    predictions = defaultdict(default_value)


    loss = 0

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            predictions['all']['total'] += labels.shape[0]
            predictions['all']['correct'] += (predicted == labels).sum().item()

            if dryrun:
                break

    for key in predictions.keys():
        if predictions[key]['total'] > 0:
            predictions[key]['avg'] = predictions[key]['correct'] / predictions[key]['total']
        else:
            predictions[key]['avg'] = float('nan')

    loss_avg = loss / (i + 1)
    return predictions, loss_avg


def _split_data(inputs, labels, source_selection='sep-half'):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    #  shuffle/sep-half/sep-1/sep-10
    if source_selection == 'shuffle':
        shuffle = torch.randperm(batch_size, device=inputs.device)
        temp_sources = inputs[shuffle].detach().clone()
        temp_true_labels = labels[shuffle].clone()
        temp_fake_label = labels
    elif source_selection == 'sep-half':
        temp_sources, inputs = inputs[:batch_size // 2], inputs[batch_size // 2:]
        temp_true_labels, labels = labels[:batch_size // 2], labels[batch_size // 2:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size // 2)
    elif source_selection == 'sep-1':
        temp_sources, inputs = inputs[0:1], inputs[1:]
        temp_true_labels, labels = labels[0:1], labels[1:]
        temp_fake_label = labels.mode(keepdim=True)[0]
    elif source_selection == 'sep-10':
        temp_sources, inputs = inputs[0:10], inputs[10:]
        temp_true_labels, labels = labels[0:10], labels[10:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(10)
    elif 'sep-p' in source_selection:
        p = int(source_selection.split('sep-p')[1])
        p_actual = int(p * batch_size / 128)
        if p_actual > batch_size or p_actual < 1:
            raise ValueError(f'Invalid sep-p option given with p={p}. Should be p in [1, 128], '
                             f'which will be scaled to the current batch size.')
        inputs, temp_sources, = inputs[0:p_actual], inputs[p_actual:]
        labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)

    else:
        raise ValueError(f'Invalid selection strategy {source_selection}.')
    return temp_sources, inputs, temp_true_labels, labels, temp_fake_label

def requires_grad(p):
    return p.requires_grad

def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    # optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())
    

    optimized_parameters = filter(requires_grad, model.parameters())

    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-sponge':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay)
    elif defs.optimizer == 'Adam':
        optimizer = torch.optim.Adam(optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    
    return optimizer, scheduler
