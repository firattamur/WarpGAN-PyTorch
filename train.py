"""

Train WarpGAN Model.

"""
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data
import torchvision.datasets as dset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import sys
from utils.commandline import load_config
from datasets.web_caricature import WebCaricatureDataset
from losses.model_losses  import AdversarialLoss, PatchAdversarialLoss
from models.model_warpgan import WarpGANGenerator, WarpGANDiscriminator

# Set random seed for reproducibility
manualSeed = 42

random.seed(manualSeed)
torch.manual_seed(manualSeed)


invNormalize = transforms.Normalize(
                                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225]
                                    )

if __name__ == "__main__":

    # load configuration file from specified configuration file path
    config = load_config(sys.argv)

    # load dataset
    dataset = WebCaricatureDataset(config)

    # create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.in_batch, shuffle=True)

    # decide which device we want to run on
    config.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    print(f"\nTraining Device: {config.device}\n")

    # load models
    warpgan_generator     = WarpGANGenerator(config).to(config.device)
    warpgan_discriminator = WarpGANDiscriminator(config).to(config.device)

    # load losses
    adversarial_loss       = AdversarialLoss(config).to(config.device)
    patch_adversarial_loss = PatchAdversarialLoss(config).to(config.device)

    # setup Adam optimizers for both G and D

    optimizerG = optim.Adam(warpgan_generator.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                            betas=(config.optimizer[1]["beta1"], config.optimizer[1]["beta2"]))

    optimizerD = optim.Adam(warpgan_discriminator.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                            betas=(config.optimizer[1]["beta1"], config.optimizer[1]["beta2"]))

    # Training Loop

    writer = SummaryWriter("runs/WarpGAN-Tensorboard")

    # For each epoch
    for epoch in range(config.num_epochs):
        
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
                    
            global_iter = epoch * len(dataloader) + i
                
            # ------------------------------------------
            # Input dicts for Generator and Disciminator
            # ------------------------------------------
                            
            generator_input_dict = {
                
                "images_photo" : data["images_photo"].to(config.device),
                "images_caric" : data["images_caric"].to(config.device),
                
                "labels_photo" : data["labels_photo"].to(config.device),
                "labels_caric" : data["labels_caric"].to(config.device),
                
                "scales_photo" : data["scales_photo"].to(config.device),
                "scales_caric" : data["scales_caric"].to(config.device),
                
            }
            
            discriminator_input_dict = {
                
                "images_photo" : data["images_photo"].to(config.device),
                "images_caric" : data["images_caric"].to(config.device),
                
                "generated_caric": None
                
            }
            
            # ------------------------------------------
            # Generator Network Forward Pass
            # ------------------------------------------
                    
            # forward pass on generator
            generator_output = warpgan_generator(generator_input_dict)
            
            # add generated caricature to discriminator input dict
            discriminator_input_dict["generated_caric"] = generator_output["generated_caric"].to(config.device)
                    
            # forward pass on discriminator
            discriminator_output = warpgan_discriminator(discriminator_input_dict)
                    
            # adversial losses on generated caricature
            
            adversial_loss_input_dict = {

                    "logits_caric" : discriminator_output["logits_caric"],
                    "logits_photo" : discriminator_output["logits_photo"],
                    "logits_generated_caric": discriminator_output["logits_generated_caric"],

                    "labels_caric" : generator_input_dict["labels_caric"],
                    "labels_photo" : generator_input_dict["labels_photo"],
                    "labels_generated_caric" : generator_input_dict["labels_photo"],

                    }
                    
            loss_DA, loss_GA = adversarial_loss(adversial_loss_input_dict)
            
            loss_DA, loss_GA = config.coef_adv * loss_DA, config.coef_adv * loss_GA
            
            # patch adversial losses on generated caricature
                    
            loss_DP, loss_GP = patch_adversarial_loss(discriminator_output["logits_caric"],
                                                      discriminator_output["logits_photo"],
                                                      discriminator_output["logits_generated_caric"])
            
            loss_DP, loss_GP = config.coef_adv * loss_DP, config.coef_adv * loss_GP
            
            # identity mapping (reconstruction) loss
                    
            loss_idt_caric = torch.mean(torch.abs(generator_output["rendered_caric"] - generator_input_dict["images_caric"]))
            loss_idt_caric = config.coef_idt * loss_idt_caric
            
            loss_idt_photo = torch.mean(torch.abs(generator_output["rendered_photo"] - generator_input_dict["images_photo"]))
            loss_idt_photo = config.coef_idt * loss_idt_photo
            
            loss_G_idt     = loss_idt_caric + loss_idt_photo
            
            # collect all losses
        
            # all losses for generator
            loss_G = loss_GA + loss_GP + loss_G_idt

            # all losses for discriminator
            loss_D = loss_DA + loss_DP

            # tensorboard writer save all losses
                        
            writer.add_scalar('Loss-Generator/Adversial',     loss_GA,        global_iter)
            writer.add_scalar('Loss-Generator/Patch',         loss_GP,        global_iter)
            writer.add_scalar('Loss-Generator/IdentityCaric', loss_idt_caric, global_iter)
            writer.add_scalar('Loss-Generator/IdentityPhoto', loss_idt_photo, global_iter)
            writer.add_scalar('Loss-Generator/Total',         loss_G,         global_iter)

            writer.add_scalar('Loss-Discriminator/Adversial', loss_DA,        global_iter)
            writer.add_scalar('Loss-Discriminator/Patch',     loss_DP,        global_iter)
            writer.add_scalar('Loss-Discriminator/Total',     loss_D,         global_iter)
            
            # reset gradients of discriminator
            warpgan_discriminator.zero_grad()
            
            # calculate gradients for discriminator
            loss_D.backward(retain_graph=True)
                    
            # reset gradients of generator
            warpgan_generator.zero_grad()
        
            # calculate gradients for generator
            loss_G.backward()
            
            # optimize generator for single step
            optimizerD.step()
            
            # optimize generator for single step
            optimizerG.step()
            
            # output training stats
            if i % config.log == 0:
                
                steps = f"[{epoch:02}/{config.num_epochs:02}][{i:04}/{len(dataloader):04}]"

                log  = steps
                log += f" G -> [Total: {loss_G:.5f}]"

                log += f"[Adversial: {loss_GA:.5f}]"
                log += f"[Patch: {loss_GP:.5f}]"
                log += f"[IdtCaric: {loss_idt_caric:.5f}]"
                log += f"[IdtPhoto: {loss_idt_photo:.5f}]\n"

                log += len(steps) * " "
                log += f" D -> [Total: {loss_D:.5f}]"
                log += f"[Adversial: {loss_DA:.5f}]"
                log += f"[Patch: {loss_DP:.5f}]"

                print(log)
                print()
                
            # check how the generator is doing by saving G's output
            if global_iter % config.summary == 0:
                
                image_photo = generator_input_dict["images_photo"][0].detach().cpu()
                image_caric = generator_input_dict["images_caric"][0].detach().cpu()

                generated   = invNormalize(generator_output["generated_caric"][0].detach().cpu())
                
                writer.add_image("Images/1 - Photo",       image_photo,     global_iter)
                writer.add_image("Images/2 - Caricature",  image_caric,     global_iter)
                writer.add_image("Images/3 - Generated",   generated,       global_iter)
