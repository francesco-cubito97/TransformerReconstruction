"""
TransReconstructor network.

The code is adapted from the METRO one https://github.com/microsoft/MeshTransformer
"""


import torch
from torch import nn
import numpy as np
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from pytorch_transformers.modeling_bert import BertLayerNorm as LayerNormClass
from manopth.manolayer import ManoLayer

from . import mano_config as cfg

class Mano(nn.Module):
    def __init__(self, args):
        super(Mano, self).__init__()

        # Right hand pkl mano model
        self.layer = ManoLayer(flat_hand_mean=args.flat_hand_mean,
                         mano_root=cfg.data_path,
                         ncomps=args.mano_ncomps,
                         use_pca=args.use_pca,
                         root_rot_mode=args.root_rot_mode,
                         joint_rot_mode=args.joint_rot_mode)

        self.vertices_num = cfg.VERT_NUM
        self.faces = self.layer.th_faces.numpy()
        self.joint_regressor = self.layer.th_J_regressor.numpy()
        
        self.joints_num = len(cfg.J_NAME)
        self.joints_name = cfg.J_NAME
        self.skeleton = cfg.SKLT_DEF
        self.root_joint_idx = cfg.ROOT_INDEX

        # Add fingertips to joint_regressor
        self.fingertip_vertex_idx = cfg.FINGERTIPS_RIGHT
        
        # One-Hot-Encoding joints vectors
        thumbtip_onehot = np.array([1 if i == self.fingertip_vertex_idx[0] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        indextip_onehot = np.array([1 if i == self.fingertip_vertex_idx[1] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        middletip_onehot = np.array([1 if i == self.fingertip_vertex_idx[2] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        ringtip_onehot = np.array([1 if i == self.fingertip_vertex_idx[3] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        pinkytip_onehot = np.array([1 if i == self.fingertip_vertex_idx[4] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        
        self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], :]
        joint_regressor_torch = torch.from_numpy(self.joint_regressor).float()
        self.register_buffer('joint_regressor_torch', joint_regressor_torch)

    # def getLayer(self):
    #     return ManoLayer(mano_root=path.join(self.mano_dir), 
    #                      flat_hand_mean=False, 
    #                      use_pca=False)
         

    def get3dJointsFromMesh(self, vertices):
        """
        This method is used to get the joint locations from the mesh
        Input:
            vertices: size = (B, 778, 3)
        Output:
            3D joints: size = (B, 21, 3)
        """
        joints = torch.einsum('bik,ji -> bjk', [vertices, self.joint_regressor_torch])
        return joints

class Trans_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(Trans_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim 

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(embeddings,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs

class Trans_Block(BertPreTrainedModel):
    '''
    The architecture of a transformer encoder block we used in METRO
    '''
    def __init__(self, config):
        super(Trans_Block, self).__init__(config)
        self.config = config
        self.bert = Trans_Encoder(config)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        self.apply(self.init_weights)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None):
        '''
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        '''
        predictions = self.bert(img_feats=img_feats, input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        if self.config.output_attentions and self.config.output_hidden_states:
            return pred_score, predictions[1], predictions[-1]
        else:
            return pred_score

class TransRecon_Network(torch.nn.Module):
    '''
    End-to-end TransRecon network for hand pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder):
        super(TransRecon_Network, self).__init__()
        self.args = args

        self.config = config
        self.backbone = backbone
        self.trans_encoder = trans_encoder

        self.cam_param_fc1 = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(cfg.VERT_SUB_NUM + cfg.JOIN_NUM, 150) 
        self.cam_param_fc3 = torch.nn.Linear(150, 3)

    def forward(self, images, mano_model, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate pose and shape template vectors
        if self.args.root_rot_mode == 'axisang':
            rot = 3
        else:
            rot = 6
    
        if not self.args.use_pca:
            self.args.mano_ncomps = 45

        self.template_pose = torch.zeros((1,  self.args.mano_ncomps + rot, 1)).to(self.args.device)
        self.template_betas = torch.zeros((1, 10, 1)).to(self.args.device)

        num_pose_params = self.template_pose.shape[1]

        # Concatenate templates and then duplicate to batch size
        
        ref_params = torch.cat([self.template_pose, self.template_betas], dim=1)
        ref_params = ref_params.expand(batch_size, -1, -1)

        # Extract global image feature using a CNN backbone
        image_feat = self.backbone(images)

        # Concatenate image feat and template parameters
        image_feat = image_feat.view(batch_size, 1, image_feat.shape[1]).expand(-1, ref_params.shape[1], -1)
        features = torch.cat([ref_params, image_feat], dim=2)

        if is_train==True:
            # Apply mask vertex/joint modeling
            # meta_masks is a tensor containing all masks, randomly generated in dataloader
            # constant_tensor is a [MASK] token, which is a floating-value vector with 0.01s
            constant_tensor = torch.ones_like(features).cuda()*0.01
            features = features*meta_masks + constant_tensor*(1 - meta_masks)     

        # Forward-pass
        if self.config.output_attentions==True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        # Get predicted parameters
        pred_pose = features[:, :num_pose_params, :]
        pred_betas = features[:, num_pose_params:, :]

        #self.args.logger.createLog("NETWORK", "Requires_grad inside network: pose={} betas={}".format(pred_pose.requires_grad, pred_betas.requires_grad))

        # Pass predicted pose and betas through MANO layer
        # to complete forward pass
        
        pred_vertices, pred_3d_joints = mano_model.layer(pred_pose.view(batch_size, pred_pose.shape[1]), pred_betas.view(batch_size, pred_betas.shape[1]))
        
        # Subsample the number of vertices to simplify the camera
        # parameters network the most possible
        pred_vertices_sub = mesh_sampler.downsample(pred_vertices)

        #self.args.logger.createLog("NETWORK", "Predicted vertices downsampled shape: {}".format(pred_vertices_sub.shape))

        predictions = torch.cat([pred_vertices_sub, pred_3d_joints], dim=1)
        
        # Learn camera parameters from predicted vertices and joints
        cam_params = self.cam_param_fc1(predictions)
        cam_params = self.cam_param_fc2(cam_params.transpose(1, 2))
        cam_params = self.cam_param_fc3(cam_params)
        cam_params = cam_params.transpose(1, 2).squeeze(-1)

        #temp_transpose = pred_vertices_sub.transpose(1, 2)
        #pred_vertices = self.upsampling(temp_transpose)
        #pred_vertices = pred_vertices.transpose(1, 2)

        if self.config.output_attentions==True:
            return (cam_params, pred_3d_joints, pred_vertices_sub,
                    pred_vertices, pred_pose, pred_betas, hidden_states, att)
        else:
            return (cam_params, pred_3d_joints, pred_vertices_sub, 
                    pred_vertices, pred_pose, pred_betas)