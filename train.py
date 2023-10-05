import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from src.dataloader import dataloader_normal
from src.utils.setup_logger import logger
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def train(model, epochs = 10, batch_size = 1,validation_split = 0.2):
    train_losses = []
    train_f1_scores = []
    train_precisions = []
    train_recalls = []

    validation_losses = []
    validation_f1_scores = []
    validation_precisions = []
    validation_recalls = []
    dataloader= dataloader_normal()

    # Calculer le nombre d'échantillons pour l'ensemble de validation
    num_samples = len(dataloader)
    num_validation = int(validation_split * num_samples)
    num_train = num_samples - num_validation

    # Diviser les données en ensembles d'entraînement et de validation
    train_dataset, validation_dataset = random_split(dataloader ,[num_train, num_validation])

    # Créer des DataLoaders pour les ensembles d'entraînement et de validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set the device for training (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, bbx, labels = data

            # Move data to the appropriate device
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            label_mapping = {"AIN": 0, "alph": 1, "baa": 2, "daa": 3, "faa": 4, "ha": 5, "Haa": 6, "Kaf": 7, "La": 8,
                             "Meme": 9, "Noun": 10, "Qua": 11, "Raa": 12, "sad": 13, "sin": 14, "Ta": 15, "waw": 16,
                             "ya": 17}
            labels = [label_mapping[label_str] for label_str in labels]
            logger.debug(labels)
            labels = torch.tensor(labels).to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                logger.debug(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}")
                running_loss = 0.0
        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        train_predicted = []  # Remplir avec les prédictions du modèle
        train_ground_truth = []  # Remplir avec les étiquettes réelles

        for data in train_dataloader:
            inputs, _, labels = data
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            labels = [label_mapping[label_str] for label_str in labels]
            labels = torch.tensor(labels).to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1).tolist()
            train_predicted.extend(predicted)
            train_ground_truth.extend(labels.tolist())

        train_f1 = f1_score(train_ground_truth, train_predicted, average='weighted')
        train_prec = precision_score(train_ground_truth, train_predicted, average='weighted')
        train_recall = recall_score(train_ground_truth, train_predicted, average='weighted')

        train_f1_scores.append(train_f1)
        train_precisions.append(train_prec)
        train_recalls.append(train_recall)

    logger.debug("Training finished")

    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(validation_dataloader):
            validation_loss = running_loss / len(validation_dataloader)
            validation_losses.append(validation_loss)

            validation_predicted = []  # Remplir avec les prédictions du modèle
            validation_ground_truth = []
            inputs, bbx, labels = data
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            label_mapping = {"AIN": 0, "alph": 1, "baa": 2, "daa": 3, "faa": 4, "ha": 5, "Haa": 6, "Kaf": 7, "La": 8,
                             "Meme": 9, "Noun": 10, "Qua": 11, "Raa": 12, "sad": 13, "sin": 14, "Ta": 15, "waw": 16,
                             "ya": 17}
            labels = [label_mapping[label_str] for label_str in labels]
            logger.debug(labels)
            labels = torch.tensor(labels).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del inputs, labels, outputs
        for data in validation_dataloader:
            inputs, _, labels = data
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            labels = [label_mapping[label_str] for label_str in labels]
            labels = torch.tensor(labels).to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1).tolist()
            validation_predicted.extend(predicted)
            validation_ground_truth.extend(labels.tolist())

        validation_f1 = f1_score(validation_ground_truth, validation_predicted, average='weighted')
        validation_prec = precision_score(validation_ground_truth, validation_predicted, average='weighted',zero_division=0)
        validation_recall = recall_score(validation_ground_truth, validation_predicted, average='weighted',zero_division=0)

        validation_f1_scores.append(validation_f1)
        validation_precisions.append(validation_prec)
        validation_recalls.append(validation_recall)
        plt.figure(figsize=(12, 6))

        # Perte (Loss)
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Entraînement')
        plt.plot(validation_losses, label='Validation')
        plt.xlabel('Époque')
        plt.ylabel('Perte (Loss)')
        plt.legend()

        # F1-score
        plt.subplot(2, 2, 2)
        plt.plot(train_f1_scores, label='Entraînement')
        plt.plot(validation_f1_scores, label='Validation')
        plt.xlabel('Époque')
        plt.ylabel('F1-score')
        plt.legend()

        # Précision (Precision)
        plt.subplot(2, 2, 3)
        plt.plot(train_precisions, label='Entraînement')
        plt.plot(validation_precisions, label='Validation')
        plt.xlabel('Époque')
        plt.ylabel('Précision (Precision)')
        plt.legend()

        # Rappel (Recall)
        plt.subplot(2, 2, 4)
        plt.plot(train_recalls, label='Entraînement')
        plt.plot(validation_recalls, label='Validation')
        plt.xlabel('Époque')
        plt.ylabel('Rappel (Recall)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
