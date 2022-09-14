"""
Start executing the training/testing/inference from here

Parts of code taken from https://github.com/microsoft/MeshTransformer
"""

import argparse
import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertConfig
from torchvision.models import mobilenet_v3_small#, MobileNet_V3_Small_Weights

from network.exec import run, runEvalAndSave
from utils.logger import Logger
from utils.mesh import Mesh
from utils.utils import createDir, setSeed
from utils.render import Renderer
from network.network import Trans_Block, TransRecon_Network, Mano 
from dataset.datamaker import make_hand_data_loader

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", default='train', type=str, required=False,
                        help="Choose one of the following types of run: train, test")

    #-----------------------------INITIALIZATION-----------------------------#
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")

    #------------------------------DATA ARGUMENT------------------------------#
    parser.add_argument("--dataset_folder", default='dataset', type=str, required=False,
                        help="Folder containing dataset")
    parser.add_argument("--train_yaml", default='freihand/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--eval_yaml", default='freihand/test.yaml', type=str, required=False,
                        help="Yaml file with all data for evaluation.")
    parser.add_argument("--num_workers", default=2, type=int, 
                        help="Workers in dataloader.")       
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.")  

    #-----------------------------CHECKPOINTS-----------------------------#
    parser.add_argument("--model_config", default='data/bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained transformer model configuration.")
    parser.add_argument("--saved_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--iteration_restart", default=0, type=int, required=False, 
                        help="Iteration to restart training phase")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--logging_steps", default=100, type=int)
    
    #---------------------------TRAINING PARAMS---------------------------#
    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, 
                        help="The initial learning rate.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs")
    parser.add_argument("--epoch_to_stop", default=30, type=int, help="Number of epochs before stopping")
    parser.add_argument("--vertices_loss_weight", default=1.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=1.0, type=float)
    parser.add_argument("--pose_loss_weight", default=1.0, type=float)
    parser.add_argument("--betas_loss_weight", default=1.0, type=float)  
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")

    #---------------------------MODEL PARAMS---------------------------#
    parser.add_argument("--use_pca", default=False, type=bool, help="Give axis-angle or rotation matrix as inputs or use PCA coefficients")
    parser.add_argument("--flat_hand_mean", default=False, type=bool, help="Use flat hand as mean instead of average hand pose")
    parser.add_argument('--root_rot_mode', default='axisang', choices=['rot6d', 'axisang'])
    parser.add_argument('--joint_rot_mode', default='axisang', choices=['rotmat', 'axisang'], help="Joint rotation inputs")
    parser.add_argument('--mano_ncomps', default=45, type=int, help="Number of PCA components")
    
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='1025,256,64', type=str, 
                        help="Input image feature dimensions")          
    parser.add_argument("--hidden_feat_dim", default='512,128,32', type=str, 
                        help="Hidden image freature dimensions")   

    parser.add_argument("--multiscale_inference", default=False, type=bool)
    parser.add_argument("--rot", default=0.0, help="Rotation for multiscale inference")
    parser.add_argument("--sc", default=1.0, help="Scale for multiscale inference")
    return parser.parse_args()

def main(args):
    # Initial setup
    args.device = torch.device(args.device)
    setSeed(args.seed)

    createDir(args.output_dir)
    args.logger = Logger("TransReconstructor", args.output_dir)

    # Visualize arguments passed
    #args.logger.createLog("MAIN", "Arguments passed: {}".format(args))

    # Mesh and SMPL utils
    mano_model = Mano(args).to(args.device)
    mano_model.layer = mano_model.layer.to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=mano_model.faces)

    # Load pretrained model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    # The final output of transformers will be the vector of pose+shape parameters
    # to pass throught the Mano layer
    output_feat_dim = input_feat_dim[1:] + [1]
    
    if args.type=="test" and args.saved_checkpoint!=None and args.saved_checkpoint!='None':
        args.logger.createLog("MAIN", "Evaluation: Loading from checkpoint {}".format(args.saved_checkpoint))
        _network = torch.load(args.saved_checkpoint)

    else:
        # Init a series of transformers blocks
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Trans_Block
            config = config_class.from_pretrained(args.model_config)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*4)

            # Update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for param in update_params:
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    args.logger.createLog("MAIN", "Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            args.logger.createLog("MAIN", "Init model from scratch.")
            trans_encoder.append(model)
        
        # Adding backbone
        args.logger.createLog("MAIN", "Using pre-trained model 'MobileNetV3'")
        #backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        backbone = mobilenet_v3_small(pretrained=True)
        # Remove the classification module
        classifier = list(backbone.classifier.children())[:-1]
        backbone.classifier = nn.Sequential(*classifier)

        # Compose the final neural network
        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        args.logger.createLog("MAIN", "Transformers total parameters: {}".format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        args.logger.createLog("MAIN", "Backbone total parameters: {}".format(backbone_total_params))

        _network = TransRecon_Network(args, config, backbone, trans_encoder)
        
        if args.saved_checkpoint!=None and args.saved_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            args.logger.createLog("MAIN", "Loading state dict from checkpoint {}".format(args.saved_checkpoint))
            checkpoint = torch.load(args.saved_checkpoint, map_location=torch.device('cpu'))
            _network.load_state_dict(checkpoint, strict=False)
            del checkpoint
        
    _network.to(args.device)
    args.logger.createLog("MAIN", "Training parameters {}".format(args))

    if args.type=="test":
        val_dataloader = make_hand_data_loader(args, 
                                               args.val_yaml, 
                                               s_train=False, 
                                               scale_factor=args.img_scale_factor)
       
        runEvalAndSave(args, 'freihand', val_dataloader, 
                          _network, mano_model, renderer, mesh_sampler)

    else:
        train_dataloader = make_hand_data_loader(args, 
                                                 args.train_yaml, 
                                                 is_train=True, 
                                                 scale_factor=args.img_scale_factor)
        
        run(args, train_dataloader, _network,
            mano_model, renderer, mesh_sampler)

if __name__ == "__main__":
    args = parseArguments()
    main(args)
