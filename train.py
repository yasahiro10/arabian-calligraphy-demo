import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.models import VGG16
from src.dataloader import dataloader_augmented, dataloader_normal
from src.utils.setup_logger import logger

# num_classes = 18
# num_epochs = 20
# batch_size = 16
# learning_rate = 0.005
#
# model = VGG16(num_classes)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
# dataset = dataloader_augmented(["Rotation","Gaussian blur","ColorJitter","GaussianNoise"],0,23,(106.67, 106.67),(2,2),4,1,0.1)
# total_step= len(dataset.data["cropped_bbox"])
# for epoch in range(num_epochs):
#     for i in range(len(dataset.data["cropped_bbox"])):
#         images = dataset.data["cropped_bbox"][i]
#         labels = dataset.data["label"][i]
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#



def train(model, epochs = 10, batch_size = 1):
    dataloader = DataLoader(dataloader_normal(), batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set the device for training (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            inputs, bbox, labels = data

            # Move data to the appropriate device
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            label_mapping = {"AIN": 0, "alph": 1, "baa": 2, "daa":3 , "faa":4,"ha":5,"haa":6,"kaf":7,"La":8,"Meme":9,"Noun":10,"Qua":11,"Raa":12,"sad":13,"sin":14,"Ta":15,"waw":16,"ya":17}
            labels = [label_mapping[label_str] for label_str in labels]
            #labels = torch.tensor(labels)
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

    logger.debug("Training finished")

    # # Validation
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in valid_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         del images, labels, outputs
    #
    #     print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
