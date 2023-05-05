import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from math import log10

from FDL.Views_Params import DiscreteApertureFSParams, AllFocusParams
from Dataloaders.Dataloader import LightFieldDataset
from Utils.Light_Field_Utils import pad_lf
from Utils.Fourier_Utils import gen_frequency_grid

'''
---------------------------------------------------------------
CLASS Trainer
    Trainer of a FDL pytorch model to reconstruct a light field
---------------------------------------------------------------
'''
class Trainer():
    def __init__(self, config, training_dataset, validation_dataset, model_name):
        super(Trainer,self).__init__()
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Read a text file line by line '''    
    def file_directory_list(self, file):
        file = open(file, 'r')
        lines = file.read().splitlines()
        return lines
    
    ''' Load the FDL optimization solver mentioned in the configuration file '''    
    def load_solver(self):
        print("\nLOADING SOLVER")
        print("USING",self.config["solver"]["name"],"SOLVER")
        if self.config["solver"]["name"] == "UnrolledADMMFDL":
            from Solvers.Unrolled_ADMM_FDL import UnrolledADMMFDL
            return UnrolledADMMFDL(nb_channels=self.config["data"]["nb_channels"], nb_iteration=self.config["solver"]["nb_iteration"], nb_fdl=self.config["task"]["nb_fdl"], rho_initial_val=self.config["solver"]["rho"], device=self.device)
        else:
            print("Unknown solver")
            import sys
            sys.exit()
    
    ''' Load a pre-trained model for the FDL optimization solver '''
    def load_model(self, solver):
        print("\nLOADING MODEL")
        start_epoch = 0
        if os.path.exists(os.path.join("Models/"+self.model_name+".ckpt")):
            print("USING PRETRAINED MODEL IN:",os.path.join("Models/"+self.model_name+".ckpt"))
            if self.device != 'cpu':
                saved_dict = torch.load(os.path.join("Models/"+self.model_name+".ckpt"))
            else:
                saved_dict = torch.load(os.path.join("Models/"+self.model_name+".ckpt"), map_location='cpu')
            start_epoch = saved_dict['epoch']
            solver.load_state_dict(saved_dict['solver_state_dict'])
        else:
            print("CREATING MODEL IN:",os.path.join("Models/"+self.model_name+".ckpt"))  
            
        return solver, start_epoch
    
    ''' Load the training and validaiton light field datasets '''
    def load_dataset(self, nb_views):
        print("\nLOADING DATASET")
        if self.config["training"]["patch_size"] > 0:
            patch_transform = transforms.Compose([transforms.RandomCrop(self.config["training"]["patch_size"])])
        else:
            patch_transform = transforms.Compose([])

        training_dataset = self.file_directory_list(self.training_dataset)
        validation_dataset = self.file_directory_list(self.validation_dataset)
        print("USING TRAINING DATASETS IN:")
        for path in training_dataset:
            print(path)
        print("USING VALIDATION DATASETS IN:")
        for path in validation_dataset:
            print(path)

        training_dataset = LightFieldDataset(directories=training_dataset,transform=patch_transform, angular_dims=[nb_views,nb_views])
        training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=int(self.config["training"]["batch_size"]), shuffle=True, drop_last=True, num_workers=8)

        validation_dataset = LightFieldDataset(directories=validation_dataset,transform=patch_transform, angular_dims=[nb_views,nb_views])
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=int(self.config["training"]["batch_size"]), shuffle=False, drop_last=False, num_workers=8)
        
        return training_dataloader,validation_dataloader
    
    ''' prepare the solver, the input, and the groundtruth for the forward pass '''
    def prepare_inputs(self, solver, views_params, batch, u, v, pad):
        gt = pad_lf(batch[0], pad, mode='reflect', use_window=True).to(self.device)
        inputs = views_params.gen_views_from_light_field(gt, u, v)
        views_fft = torch.fft.fftn(inputs, dim=(-2, -1))
        n_freq_x = views_fft.shape[-1]
        n_freq_y = views_fft.shape[-2]
        solver.fdl_model.w_x, solver.fdl_model.w_y = gen_frequency_grid(n_freq_x, n_freq_y, half_x_dim=True, centered_zero=True)
        solver.fdl_model.to(self.device)
        fdl = torch.zeros(inputs.shape[:2] + solver.d_fdl.shape[-1:] + inputs.shape[-2:],device=inputs.device, dtype=inputs.dtype)
        
        return solver, gt, inputs, fdl
    
    ''' prepare the output and the groundtruth for the loss computation '''    
    def prepare_outputs(self, solver, render_params, fdl, gt, u, v, pad):
        output = solver.generate_views(fdl, render_params)
        output = output.unflatten(2, u.shape)
        output = torch.fft.ifftn(output, dim=(-2, -1)).real
        output = output[..., pad[2]:output.shape[-2] - pad[3], pad[0]:output.shape[-1] - pad[1]]
        gt = gt[..., pad[2]:gt.shape[-2] - pad[3], pad[0]:gt.shape[-1] - pad[1]]
        
        return gt, output
    
    ''' set the rng seed to have consistent output '''
    def set_rng(self, seed=0):
        rng_cpu = torch.get_rng_state()
        rng_gpu = None
        if torch.cuda.is_available():
            rng_gpu = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)
        
        return rng_cpu, rng_gpu
        
    ''' reset a pre-set rng seed '''       
    def reset_rng(self, rng_cpu, rng_gpu):
        torch.set_rng_state(rng_cpu)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_gpu)
     
    ''' save the model weights of the FDL optimization solver '''     
    def save_model(self, solver, epoch, optimizer, scheduler, best_average_error, ave_error_valid):
        torch.save({'solver_state_dict': solver.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, os.path.join("Models",self.model_name + ".ckpt"))
        if best_average_error < ave_error_valid or best_average_error == -1:
            best_average_error = ave_error_valid
            print("Best model!")
            torch.save({'solver_state_dict': solver.state_dict()}, os.path.join("Models",self.model_name + "_best.ckpt"))
        
        return best_average_error
    
    ''' Compute the training of a FDL optimization model '''
    def train(self):
        torch.cuda.empty_cache()
        ################# PARAMETERS #################
        nb_views = self.config["data"]["angular_dims"]
        nb_measurements = self.config["data"]["number_of_shots"]
        v, u = torch.meshgrid(torch.arange(float(int(nb_views/2.0)), -float((int(nb_views/2.0)+1)), -1), torch.arange(-float(int(nb_views/2.0)), float(int(nb_views/2.0)+1)))
        v = v.to(self.device)
        u = u.to(self.device)
        d_fs = torch.tensor(self.config["task"]["disparity_focal_stack"])
        pad = (8, 8, 8, 8)

        views_params = DiscreteApertureFSParams(u, v, d_fs)
        render_params = AllFocusParams(u, v)
        
        ################# LOAD SOLVER #################
        solver = self.load_solver().to(self.device)
        
        ################# LOAD MODEL #################
        solver, start_epoch = self.load_model(solver)
                        
        ################# LOAD OPTIMIZER #################
        optimizer = optim.Adam(params=solver.parameters(), lr=float(self.config["training"]["learning_rate"]))
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=0.1)
        if os.path.exists(os.path.join("Models",self.model_name,".ckpt")):
            optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
            scheduler.load_state_dict(saved_dict['scheduler_state_dict'])

        loss_function = torch.nn.MSELoss()
        
        ################# LOAD DATASET #################
        training_dataloader, validation_dataloader = self.load_dataset(nb_views)
        
        ################# START TRAINING #################
        print("\nSTARTING TRAINING")
        best_average_error = -1
        for epoch in range(start_epoch, self.config["training"]["epochs"]):
            print("\nEpoch", epoch, ":")
            train_results = []
            valid_results = []
            
            nb_batch = 1.0*training_dataloader.__len__()
            patch_done = 0
            
            ################# TRAINING PHASE #################
            print("TRAINING PHASE")
            for ii, batch in enumerate(training_dataloader):
                optimizer.zero_grad()
                
                solver, gt, inputs, fdl = self.prepare_inputs(solver, views_params, batch, u, v, pad)
                output = solver(fdl, inputs, views_params)
                gt, output = self.prepare_outputs(solver, render_params, output, gt, u, v, pad)

                loss_value = loss_function(output, gt) 
                loss_value.backward()
                optimizer.step()
                if loss_value != 0:
                    psnr = 10 * log10(1/loss_value.item())
                else:
                    psnr = 0
                train_results.append(psnr)
                
                patch_done += 1
                print("[" + str(int(patch_done)) + "/" + str(int(nb_batch)) + "]","- PSNR:",str(round(psnr,2)) + "db","- Mean PSNR:", str(round(np.mean(train_results),2)) + "db","- Memory:", str(torch.cuda.max_memory_allocated(self.device)/(10**9)) + "Gb")

            ################# VALIDATION PHASE #################
            print("VALIDATION PHASE")
            
            nb_batch = 1.0*validation_dataloader.__len__()
            patch_done = 0
            
            rng_cpu, rng_gpu = self.set_rng()
            
            for ii, batch in enumerate(validation_dataloader):
                solver, gt, inputs, fdl = self.prepare_inputs(solver, views_params, batch, u, v, pad)
                with torch.no_grad():
                    output = solver(fdl, inputs, views_params)
                gt, output = self.prepare_outputs(solver, render_params, output, gt, u, v, pad)

                loss_value = loss_function(output, gt) 
                if loss_value != 0:
                    psnr = 10 * log10(1/loss_value.item())
                else:
                    psnr = 0
                valid_results.append(psnr)
                
                patch_done += 1
                print("[" + str(int(patch_done)) + "/" + str(int(nb_batch)) + "]","- PSNR:",str(round(psnr,2)) + "db","- Mean PSNR:", str(round(np.mean(valid_results),2)) + "db","- Memory:", str(torch.cuda.max_memory_allocated(self.device)/(10**9)) + "Gb")
            
            self.reset_rng(rng_cpu, rng_gpu)
            
            ################# PERFORMANCES #################
            ave_error_train = sum(train_results) / len(train_results)
            ave_error_valid = sum(valid_results) / len(valid_results)
            
            print()
            print("Average training PSNRs:",ave_error_train, "db")
            print("Average validation PSNRs:",ave_error_valid, "db")
            print()
            
            ################# SAVE MODEL #################
            best_average_error = self.save_model(solver, epoch, optimizer, scheduler, best_average_error, ave_error_valid)
    
    ''' Launch the training '''    
    def forward(self):
        self.train()