import argparse
from torch import optim
from data.dataloader import *
from modules.vae import *
from modules.train import *


def main(args):

    # initialise CUDA
    if args.cuda:
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # instantiate models
    if args.network_type == "neuralODE":

        structural_estimator_model = ODEnet(3600, 14, int((3600 + 14) / 2), num_blocks=args.num_blocks_estim,
                                            tol=args.tol, dropout=args.dropout_estim)
        structural_estimator_optimizer = optim.Adam(structural_estimator_model.parameters(), lr=args.lr)

        energetic_estimator_model = ODEnet(3600, 1, int(3600 / 2), num_blocks=args.num_blocks_estim, tol=args.tol,
                                           dropout=args.dropout_estim)
        energetic_estimator_optimizer = optim.Adam(energetic_estimator_model.parameters(), lr=args.lr)

        predictor_model = ODEnet(args.latent_dim, 1, int((args.latent_dim) / 2), num_blocks=args.num_blocks_pred,
                                 dropout=args.dropout_pred)
        predictor_optimizer = optim.Adam(predictor_model.parameters(), lr=args.lr)

        vae_model = FlowVAE(input_dim=3600, hidden_dim=200, latent_dim=args.latent_dim, num_blocks=args.num_blocks_vae,
                            num_flows=args.num_flows, dropout=args.dropout_vae, gauss_mix=args.gauss_mix,
                            num_gauss=args.num_gauss, network='odenet', tol=args.tol).to(device)
        vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-5)

    elif args.network_type == "ResNet":

        structural_estimator_model = ResidualNet(3600, 14, int((3600 + 14) / 2), num_blocks=args.num_blocks_estim,
                                                 dropout=args.dropout_estim)
        structural_estimator_optimizer = optim.Adam(structural_estimator_model.parameters(), lr=args.lr)

        energetic_estimator_model = ResidualNet(3600, 1, int(3600 / 2), num_blocks=args.num_blocks_estim,
                                                dropout=args.dropout_estim)
        energetic_estimator_optimizer = optim.Adam(energetic_estimator_model.parameters(), lr=args.lr)

        predictor_model = ResidualNet(args.latent_dim, 1, int((args.latent_dim) / 2), num_blocks=args.num_blocks_pred,
                                      dropout=args.dropout_pred)
        predictor_optimizer = optim.Adam(predictor_model.parameters(), lr=args.lr)

        vae_model = FlowVAE(input_dim=3600, hidden_dim=200, latent_dim=args.latent_dim, num_blocks=args.num_blocks_vae,
                            num_flows=args.num_flows, dropout=args.dropout_vae, gauss_mix=args.gauss_mix,
                            num_gauss=args.num_gauss, network='resnet').to(device)
        vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-5)

    # load data
    train_loader, val_loader_1, val_loader_2 = material_loader(args.data_loc, args.batch_size)

    # train
    struc_estim_train_loss_list = []
    struc_estim_test_loss_list = []
    for epoch in range(1, args.epochs_estim + 1):
        train_loss = structural_estimator_train(epoch, structural_estimator_model, structural_estimator_optimizer, train_loader, args.log_interval, device)
        test_loss = structural_estimator_test(epoch, structural_estimator_model, structural_estimator_optimizer, val_loader_1, args.log_interval, device)
        struc_estim_train_loss_list.append(train_loss.cpu().detach().numpy())
        struc_estim_test_loss_list.append(test_loss.cpu().detach().numpy())
    print('structural estimator training done')

    energ_estim_train_loss_list = []
    energ_estim_test_loss_list = []
    for epoch in range(1, args.epochs_estim + 1):
        train_loss = energetic_estimator_train(epoch, energetic_estimator_model, energetic_estimator_optimizer, train_loader, args.log_interval, device)
        test_loss = energetic_estimator_test(epoch, energetic_estimator_model, energetic_estimator_optimizer, val_loader_1, args.log_interval, device)
        energ_estim_train_loss_list.append(train_loss.cpu().detach().numpy())
        energ_estim_test_loss_list.append(test_loss.cpu().detach().numpy())
    print('energy estimator training done')

    vae_train_loss_list = []
    vae_test_loss_list = []
    latent_list = []
    for epoch in range(1, args.epochs_vae + 1):
        train_loss, latent = vae_train(epoch, vae_model, vae_optimizer, train_loader, args.log_interval, device)
        test_loss = vae_test(epoch, vae_model, vae_optimizer, val_loader_1, args.log_interval, device)
        vae_train_loss_list.append(train_loss.cpu().detach().numpy())
        vae_test_loss_list.append(test_loss.cpu().detach().numpy())
        latent_list.append(latent)
    print('latent space training done')

    pred_train_loss_list = []
    pred_test_loss_list = []
    pred_latent_list = []
    for epoch in range(1, args.epochs_pred + 1):
        train_loss, latent = pred_train(epoch, structural_estimator_model, energetic_estimator_model, vae_model, predictor_model, predictor_optimizer, val_loader_1, args.log_interval, device)
        test_loss = pred_test(epoch, structural_estimator_model, energetic_estimator_model, vae_model, predictor_model, predictor_optimizer, val_loader_2, args.log_interval, device)
        pred_train_loss_list.append(train_loss.cpu().detach().numpy())
        pred_test_loss_list.append(test_loss.cpu().detach().numpy())
        pred_latent_list.append(latent)
    print('predictor training done')

    # save results
    np.save('struc_estim_train_loss_list', struc_estim_train_loss_list)
    np.save('struc_estim_test_loss_list', struc_estim_test_loss_list)
    np.save('energ_estim_train_loss_list', energ_estim_train_loss_list)
    np.save('energ_estim_test_loss_list', energ_estim_test_loss_list)
    np.save('vae_train_loss_list', vae_train_loss_list)
    np.save('vae_test_loss_list', vae_test_loss_list)
    np.save('latent_list', latent_list)
    np.save('pred_train_loss_list', pred_train_loss_list)
    np.save('pred_test_loss_list', pred_test_loss_list)
    np.save('pred_latent_list', pred_latent_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="train/test batch size", type=int, default=256)
    parser.add_argument("--epochs_estim", help="epochs for estimator network", type=int, default=6)
    parser.add_argument("--epochs_vae", help="epochs for VAE", type=int, default=600)
    parser.add_argument("--epochs_pred", help="epochs for predictor network", type=int, default=300)
    parser.add_argument("--seed", help="random seed", type=int, default=9)
    parser.add_argument("--log_interval", help="number batches to wait before logging training status", type=int,
                        default=10)
    parser.add_argument("--latent_dim", help="VAE latent space dimension", type=int, default=15)
    parser.add_argument("--gauss_mix", help="uses Gaussian mixture prior if specified", default=True, action='store_false')
    parser.add_argument("--num_gauss", help="number of Gaussian components in mixture prior", type=int, default=14)
    parser.add_argument("--num_flows", help="number of spline autoregressive flows to use in VAE", type=int, default=3)
    parser.add_argument("--num_blocks_estim", help="number of residual/ODE blocks to use in estimator network",
                        type=int, default=3)
    parser.add_argument("--num_blocks_pred", help="number of residual/ODE blocks to use in VAE", type=int, default=3)
    parser.add_argument("--num_blocks_vae", help="number of residual/ODE blocks to use in predictor network", type=int,
                        default=3)
    parser.add_argument("--dropout_estim", help="dropout probability in estimator network", type=int, default=0.3)
    parser.add_argument("--dropout_pred", help="dropout probability in predictor network", type=int, default=0.3)
    parser.add_argument("--dropout_vae", help="dropout probability in VAE encoder/decoder", type=int, default=0.3)
    parser.add_argument("--tol", help="ODE solver tolerance", type=int, default=1e-3)
    parser.add_argument("--lr", help="learning rate", type=int, default=1e-5)
    parser.add_argument("--data_loc", help="data location on drive", default=r'/home/taymaz/Downloads/MP.npy')
    parser.add_argument("--network_type", help="ResNet or neuralODE", default="neuralODE", choices=['ResNet', 'neuralODE'])
    parser.add_argument("--cuda", help="use CUDA acceleration", default=True, action='store_false')
    args = parser.parse_args()

    main(args)
