import torch.utils.data
from modules.loss import *


# define epoch based structural trainer
def structural_estimator_train(epoch, structural_estimator_model, structural_estimator_optimizer, train_loader, log_interval, device):
    structural_estimator_model.train()
    for batch_idx, (data, tar_energy, tar_structure) in enumerate(train_loader):
        data = data.to(device)
        tar = tar_structure.view(-1, 1).to(torch.long).to(device)
        structural_estimator_optimizer.zero_grad()
        tar_pred = structural_estimator_model(data)
        loss = pred_loss_CE(tar, tar_pred)
        loss.backward()
        structural_estimator_optimizer.step()
    return loss

# define epoch based structural tester
def structural_estimator_test(epoch, structural_estimator_model, structural_estimator_optimizer, test_loader, log_interval, device):
    structural_estimator_model.eval()
    with torch.no_grad():
        for batch_idx, (data, tar_energy, tar_structure) in enumerate(test_loader):
            data = data.to(device)
            tar = tar_structure.view(-1, 1).to(torch.long).to(device)
            structural_estimator_optimizer.zero_grad()
            tar_pred = structural_estimator_model(data)
            loss = pred_loss_CE(tar, tar_pred)
            if batch_idx % log_interval == 0:
                print(f'Epoch:{epoch}, pre-train val prediction loss: {loss.item()}')
    return loss

# define epoch based energetic trainer
def energetic_estimator_train(epoch, energetic_estimator_model, energetic_estimator_optimizer, train_loader, log_interval, device):
    energetic_estimator_model.train()
    for batch_idx, (data, tar_energy, tar_structure) in enumerate(train_loader):
        data = data.to(device)
        tar = tar_energy.view(-1, 1).to(device)
        energetic_estimator_optimizer.zero_grad()
        tar_pred = energetic_estimator_model(data)
        loss = pred_loss_MSE(tar, tar_pred)
        loss.backward()
        energetic_estimator_optimizer.step()
    return loss

# define epoch based energetic tester
def energetic_estimator_test(epoch, energetic_estimator_model, energetic_estimator_optimizer, test_loader, log_interval, device):
    energetic_estimator_model.eval()
    with torch.no_grad():
        for batch_idx, (data, tar_energy, tar_structure) in enumerate(test_loader):
            data = data.to(device)
            tar = tar_energy.view(-1, 1).to(device)
            energetic_estimator_optimizer.zero_grad()
            tar_pred = energetic_estimator_model(data)
            loss = pred_loss_MSE(tar, tar_pred)
            if batch_idx % log_interval == 0:
                print(f'Epoch:{epoch}, pre-train val prediction loss: {loss.item()}')
    return loss

# define epoch based VAE trainer
def vae_train(epoch, vae_model, vae_optimizer, train_loader, log_interval, device):
    vae_model.train()
    latent = []
    for batch_idx, (data, tar_energy, tar_structure) in enumerate(train_loader):
        data = data.to(device)
        tar_structure = tar_structure.view(-1, 1).to(torch.long).to(device)
        tar_energy = tar_energy.view(-1, 1).to(device)
        vae_optimizer.zero_grad()
        x_pred, mu, logvar, _, energetic_embed = vae_model(data, tar_structure, tar_energy)
        latent.append(energetic_embed.detach().cpu().numpy())
        loss, _, _ = vae_loss(x_pred, data, mu, logvar)
        loss.backward()
        vae_optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
    return loss, latent

# define epoch based VAE tester
def vae_test(epoch, vae_model, vae_optimizer, test_loader, log_interval, device):
    vae_model.eval()
    with torch.no_grad():
        for i, (data, tar_energy, tar_structure) in enumerate(test_loader):
            data = data.to(device)
            tar_structure = tar_structure.view(-1, 1).to(torch.long).to(device)
            tar_energy = tar_energy.view(-1, 1).to(device)
            vae_optimizer.zero_grad()
            x_pred, mu, logvar, _, _ = vae_model(data, tar_structure, tar_energy)
            loss, _, _ = vae_loss(x_pred, data, mu, logvar)
    return loss

# define epoch based predictor trainer
def pred_train(epoch, structural_estimator_model, energetic_estimator_model, vae_model, predictor_model, predictor_optimizer, train_loader, log_interval, device):
    structural_estimator_model.eval()
    energetic_estimator_model.eval()
    vae_model.eval()
    predictor_model.train()
    latent = []
    for batch_idx, (data, tar_energy, tar_structure) in enumerate(train_loader):
        data = data.to(device)
        context_structure = structural_estimator_model(data).argmax(dim=1).view(-1, 1)
        context_energy = energetic_estimator_model(data).argmax(dim=1).view(-1, 1)
        _, _, _, _, energetic_embedding = vae_model(data, context_structure, context_energy)
        tar_structure = tar_structure.view(-1, 1).to(torch.long).to(device)
        tar_energy = tar_energy.view(-1, 1).to(device)
        predictor_optimizer.zero_grad()
        pred_energy = predictor_model(energetic_embedding)
        latent.append(energetic_embedding.detach().cpu().numpy())
        loss = pred_loss_MSE(tar_energy, pred_energy)
        loss.backward()
        predictor_optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
    return loss, latent

# define epoch based predictor tester
def pred_test(epoch, structural_estimator_model, energetic_estimator_model, vae_model, predictor_model, predictor_optimizer, test_loader, log_interval, device):
    structural_estimator_model.eval()
    energetic_estimator_model.eval()
    vae_model.eval()
    predictor_model.eval()
    latent = []
    with torch.no_grad():
        for batch_idx, (data, tar_energy, tar_structure) in enumerate(test_loader):
            data = data.to(device)
            context_structure = structural_estimator_model(data).argmax(dim=1).view(-1, 1)
            context_energy = energetic_estimator_model(data).argmax(dim=1).view(-1, 1)
            _, _, _, _, energetic_embedding = vae_model(data, context_structure, context_energy)
            tar_structure = tar_structure.view(-1, 1).to(torch.long).to(device)
            tar_energy = tar_energy.view(-1, 1).to(device)
            predictor_optimizer.zero_grad()
            pred_energy = predictor_model(energetic_embedding)
            latent.append(energetic_embedding.detach().cpu().numpy())
            loss = pred_loss_MSE(tar_energy, pred_energy)
    return loss
