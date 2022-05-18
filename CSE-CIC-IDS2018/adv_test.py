import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

epsilons = [0, .05, .1, .15, .2, .25, .3]


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def get_advSamples(model, epsilon, data_source):
    data = data_source[:5000, :-1]
    label = data_source[:5000, -1]
    target = np.random.randint(0, model.size_output, len(label))
    data = data[label != target]
    target = target[label != target]
    use_gpu = torch.cuda.is_available()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=0.005)
    if use_gpu:  # 1
        model = model.cuda()
        data = torch.Tensor(data).cuda()
        target = torch.Tensor(target).cuda()
    data.requires_grad = True
    activations = model(data)

    criteria = nn.CrossEntropyLoss()
    loss = criteria(activations, target.long())
    optimizer_model.zero_grad()
    loss.backward()
    # Collect datagrad
    data_grad = data.grad.data
    activations = activations.data.cpu().numpy()
    predicts = np.argmax(activations, 1)

    adv_examples = []
    for i in range(len(target)):
        perturbed_data = fgsm_attack(data[i], epsilon, data_grad[i])
        perturbed_data = torch.unsqueeze(perturbed_data, 0)
        output = model(perturbed_data)
        final_pred = np.argmax(F.softmax(output, dim=1).data.cpu().numpy())
        if final_pred == target[i] and target[i] != predicts[i]:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append(adv_ex)
    return np.array(adv_examples)


def get_advSamples_random(model, epsilon, num):
    data = np.random.random((num, model.size_input))
    target = np.random.randint(0, model.size_output, num)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=0.005)
    use_gpu = torch.cuda.is_available()
    if use_gpu:  # 1
        model = model.cuda()
        data = torch.Tensor(data).cuda()
        target = torch.Tensor(target).cuda()

    data.requires_grad = True
    activations = model(data)
    criteria = nn.CrossEntropyLoss()
    loss = criteria(activations, target.long())
    optimizer_model.zero_grad()
    loss.backward()
    # Collect datagrad
    data_grad = data.grad.data
    activations = activations.data.cpu().numpy()
    predicts = np.argmax(activations, 1)

    adv_examples = []
    for i in range(len(target)):
        perturbed_data = fgsm_attack(data[i], epsilon, data_grad[i])
        perturbed_data = torch.unsqueeze(perturbed_data, 0)
        output = model(perturbed_data)
        final_pred = np.argmax(F.softmax(output, dim=1).data.cpu().numpy())
        if final_pred == target[i] and target[i] != predicts[i]:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append(adv_ex)
    return np.array(adv_examples)
