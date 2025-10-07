# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torch.nn import functional as F
from GroundingDINO.util.misc import inverse_sigmoid

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .context_density_module import CCM, CACCM, MLCCM
from .coutning_attention import DensityAwareEnhance
from .context_aware_module import CADM
from .counter import DensityRegressor
from .query_enhanced_module import QueryLinguishModule, QueryLinguishDensityModule
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)
from utils.tsne import vis_feature_tsn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )

        if use_text_enhancer:  # True
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:  # True
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,  # True
            use_transformer_ckpt=use_transformer_ckpt,  # True
        )

        # setting the regressor
        self.regressor = DensityRegressor(counter_dim=256)
        self.density_enhance_module = DensityAwareEnhance(channel_attention=True)

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:  # 6
                self.level_embed = nn.Parameter(
                    torch.Tensor(num_feature_levels, d_model)
                )
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt  # True
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        self.enc_output_regression = nn.Linear(d_model, d_model)
        self.enc_output_regression_norm = nn.LayerNorm(d_model)

        if two_stage_type == "no":
            self.init_ref_points(num_queries)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self.cross_attention = CrossAttentionLayer(d_model)
        self.query_enhance_module = QueryLinguishDensityModule(d_model)
        # self.query_linguish_module = QueryLinguishModule(d_model)
        print(
            f"Total added parameters for cross attention: {sum(p.numel() for p in self.cross_attention.parameters())}"
        )
        self.query_detach = False
        self.density_warmup = False
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(
        self,
        srcs,
        masks,
        refpoint_embed,
        pos_embeds,
        tgt,
        attn_mask=None,
        text_dict=None,
        img_shape=None,
    ):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi] # list of 4: [(bs, 256,100,137) (bs,256,50,69), (bs, 256,25,35), (bs, 256,13,18)]
            - masks: List of multi masks [bs, hi, wi]      # list of 4: [(bs,100,137), (bs,50,69), (bs,25,35), (bs,13,18)]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None
        memory = src_flatten
        memory_text = text_dict["encoded_text"]
        # """ encoder
        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )

        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        # """
        text_dict["encoded_text"] = memory_text
        txt_embs = text_dict["encoded_text"]
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()

        #########################################################
        # Begin Context Aware Module
        #########################################################
        # memory, memory_text = self.cadm(
        #     memory,
        #     pos=lvl_pos_embed_flatten,
        #     level_start_index=level_start_index,
        #     spatial_shapes=spatial_shapes,
        #     valid_ratios=valid_ratios,
        #     key_padding_mask=mask_flatten,
        #     memory_text=text_dict["encoded_text"],
        #     text_attention_mask=~text_dict["text_token_mask"],
        #     # we ~ the mask . False means use the token; True means pad the token
        #     position_ids=text_dict["position_ids"],
        #     text_self_attention_masks=text_dict["text_self_attention_masks"],
        # )

        #########################################################
        # End Context Aware Module
        #########################################################
        ## TODO: adding the text-guide mask
        output_memory_regression, output_proposals = gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        output_memory_regression = self.enc_output_regression_norm(
            self.enc_output_regression(output_memory_regression)
        )
        # 使用enc_out_embed to calculate the similarity of text and vision features.
        if text_dict is not None:
            enc_outputs_class_unselected = self.enc_out_class_embed(
                output_memory_regression, text_dict
            )
        else:
            enc_outputs_class_unselected = self.enc_out_class_embed(
                output_memory_regression
            )
        context_aware_similarity_feature = (
            output_memory_regression * text_dict["encoded_text"][:, 0:1, :]
        )
        output_memory_regression_dict = {
            "vision_feature": output_memory_regression,
            "spatial_shape": spatial_shapes,
        }
        N_, S_, C_ = memory.shape
        _cur = 0
        cas_feature = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # mask_flatten_ = mask_flatten[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            feature_flatten_ = context_aware_similarity_feature[
                :, _cur : (_cur + H_ * W_)
            ].view(N_, H_, W_, -1)
            cas_feature.append(feature_flatten_.permute(0, 3, 1, 2))
            _cur += H_ * W_
        # get the min similarity map
        # b, l, d = enc_outputs_class_unselected.shape
        # similarity_logits2 = torch.full((b, l), float('inf'), dtype=enc_outputs_class_unselected.dtype, device=enc_outputs_class_unselected.device)
        # valid_text = torch.sum(text_dict['text_subject_mask'],dim=1)
        # for i in range(b):
        #     v = valid_text[i]
        #     min_val = torch.min(enc_outputs_class_unselected[i, :, :v], dim=1, keepdim=True).values
        #     similarity_logits2[i, :] = min_val.squeeze(-1)

        # similarity_logits = enc_outputs_class_unselected[:, :, 0]  # (bs, \sum{hw})
        # because 0 is [CLS], which can stand for the whole caption
        ## TODO: 更改这个部分，这边只使用cls token用来获取文本信息，显然是不够的，我们可以增强这个文本的使用。
        ## 我们需要在这个地方增强差异化，首先需要在encoder output这个部分增加contrastive learning，增强视觉特征的分类。
        ## 在这个地方计算相似度的时候，不能只拿cls token，因为cls token的话，你要是拿去分类，是很难得到低分的，所以我们需要改进一下。
        ## 1. 取cls token和sentence里面的最低分，相乘。实现一票否决权。只要有一个attribute不符合，就可以得到低分。
        ## 2. 监督visual和sentence不同的部分有不同的响应分数，对于class高响应，对于不符合的attribute低响应，从而监督文本和视觉特征。
        ## 并且细化文本和视觉特征到attribute层面。
        ## additional：我们认为文本特征是划分类别的，比如可以构成一个整体的，他们之间的关联度高，他们之间就有高相似度，比如white car on the floor
        ## 其中white car有高相似度，相比之下，on the floor有高相似度。因此我们取其中一个的最低分就可以表示所有的attribute的最低分。

        """
        TODO: 1. 需要变量： 1) the length of text tokens
        2) find the min score to get the unlogit similarity map (from a batch)
            if training
            a. if batch size == 1
                get the positive similarity map 
            b. if batch size > 1
                get the positive and negative similarity map
            else:
                get the positive similarity map
                
        3) return the similarity logits
        
        3) adding the contrastive learning loss
            a. get the positive and visual feature by the point map 
            b. get the contrastive loss
        
        """
        similarity_logits = enc_outputs_class_unselected[:, :, 0]  # (bs, \sum{hw})

        N_, S_ = similarity_logits.shape
        _cur = 0
        sim_maps = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # mask_flatten_ = mask_flatten[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            sim_map = similarity_logits[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, -1)
            sim_maps.append(sim_map.permute(0, 3, 1, 2).sigmoid())
            _cur += H_ * W_

        N_, S_, C_ = memory.shape
        _cur = 0
        cnn = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # mask_flatten_ = mask_flatten[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            memory_flatten_ = memory[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, -1)
            cnn.append(memory_flatten_.permute(0, 3, 1, 2))
            _cur += H_ * W_
        ## TODO: Density Regression
        density, density_hidden_features = self.regressor(
            cas_feature, cnn, img_shape=img_shape, hidden_output=True
        )

        # pred_num_reg = torch.sum(density, dim=[1,2,3]) / 60
        if self.density_warmup:
            ## TODO: Density Feature Attention Enhance Module
            cnn = cnn
            print("wrong")
        else:
            # from utils.tsne import vis_feature_tsn
            # for i in range(len(cnn) - 2):
            #     for j in range(cnn[0].shape[0]):
            #         vis_feature_tsn(cnn[i][j:j+1,...], f'exp/visualization_experiment/feature/SDensityGD/visual_features/density_{i}_{j}_feature.jpg', size=32 * (2-i), dim=3)
            cnn = self.density_enhance_module(cnn, density_hidden_features)
            # for i in range(len(cnn) - 2):
            #     for j in range(cnn[0].shape[0]):
            #         vis_feature_tsn(cnn[i][j:j+1,...], f'feature_vis/density_{i}_{j}_feature.jpg', size=64 * (2-i), dim=3)
        memory_flatten = []
        for memory_spatial in cnn:
            b, c, h, w = memory_spatial.shape
            memory_flatten_ = memory_spatial.view(b, c, -1).permute(0, 2, 1)
            memory_flatten.append(memory_flatten_)
        memory = torch.cat(memory_flatten, dim=1)

        ### decoder part
        # 将得到的output_memory经过一个linear进行变形。
        if self.two_stage_type == "standard":
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            # 使用enc_out_embed to calculate the similarity of text and vision features.
            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(
                    output_memory, text_dict
                )
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            # because 0 is [CLS], which can stand for the whole caption
            topk_logits = enc_outputs_class_unselected[:, :, 0]  # (bs, \sum{hw})
            # use 3 layer MLP to get the relatvie coord for every one.
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            # ### temp mod
            # topk = 500

            # get the relative score topk
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # lower_idxes, higher_idxes = split_tokens(topk_proposals)
            # lower_tokens = torch.gather( output_memory, 1, lower_idxes.unsqueeze(-1).expand(-1, -1, self.d_model))
            # higher_tokens = torch.gather(output_memory, 1, higher_idxes.unsqueeze(-1).expand(-1, -1, self.d_model))

            # text_subject_mask = text_dict["text_subject_mask"]
            # text_context_mask = text_dict["text_context_mask"]

            # # Extracting the tokens using the mask
            # subject_text_tokens = [text[mask] for text, mask in zip(text_dict["encoded_text"], text_subject_mask)]
            # context_text_tokens = [text[mask] for text, mask in zip(text_dict["encoded_text"], text_context_mask)]
            # max_size = text_dict["encoded_text"].size(1)
            # padded_subject_text_tokens = torch.stack([F.pad(t, (0, 0, 0, max_size - t.size(0))) for t in subject_text_tokens])
            # padded_context_text_tokens = torch.stack([F.pad(t, (0, 0, 0, max_size - t.size(0))) for t in context_text_tokens])

            # subject_mask = torch.stack([torch.cat([torch.ones(t.size(0)).to(t.device), torch.zeros(max_size - t.size(0)).to(t.device)]) for t in subject_text_tokens])
            # context_mask = torch.stack([torch.cat([torch.ones(t.size(0)).to(t.device), torch.zeros(max_size - t.size(0)).to(t.device)]) for t in context_text_tokens])

            # lower_tokens = self.cross_attention(lower_tokens, padded_subject_text_tokens, V_mask=subject_mask)
            # if context_mask.sum() > 2: # not all [CLS][SEP] tokens
            #     higher_tokens = self.cross_attention(higher_tokens, padded_context_text_tokens, V_mask=context_mask)
            # # 通过cross-attention更新了low-level的特征信息，而后再将其放到原来的output-memory中，做了特征的更新。
            # updated_lower_tokens = self.cross_attention(lower_tokens, higher_tokens)
            # output_memory = output_memory.scatter(1, lower_idxes.unsqueeze(-1).expand(-1, -1, 256), updated_lower_tokens)

            # 获取选择的topk的proposal的坐标位置，init-box-proposals是原始的初始化的坐标位置，ref则是增强之后的结果。
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )  # unsigmoid

            density_feature_flatten = []
            for density_hidden_feature in density_hidden_features:
                b, c, h, w = density_hidden_feature.shape
                density_flatten_ = density_hidden_feature.view(b, c, -1).permute(
                    0, 2, 1
                )
                density_feature_flatten.append(density_flatten_)
            density_features = torch.cat(density_feature_flatten, dim=1)

            density_features_topk = torch.gather(
                density_features,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )

            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid
            # 获取对应位置的特征信息。
            tgt_undetach = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )
            # 使用了可学习的target embedding
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(self.two_stage_type)
            )

        img_embs = tgt

        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, nq, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        tgt_enhanced = tgt
        # if self.density_warmup:
        #     tgt_enhanced = tgt
        # else:
        #     if self.query_detach:
        #         density_feature_topk_detach = density_features_topk.detach()
        #         tgt_enhanced = self.query_enhance_module(tgt, text_dict["encoded_text"],density_feature_topk_detach)
        #     else:
        #         tgt_enhanced = self.query_enhance_module(tgt, text_dict["encoded_text"],density_features_topk)

        # tgt_enhanced = self.query_linguish_module(tgt, text_dict["encoded_text"])
        # tgt_linguish_density = self.query_linguish_density_module(tgt, text_dict["encoded_text"], density_features_topk)
        # tgt_enhanced = tgt + tgt_enhanced
        hs, references = self.decoder(
            tgt=tgt_enhanced.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model # list of 6
        # references: n_dec+1, bs, nq, query_dim # list of 7
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":  # yes
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return (
            hs,
            references,
            hs_enc,
            ref_enc,
            init_box_proposal,
            img_embs,
            txt_embs,
            density,
            output_memory_regression_dict,
            sim_maps,
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share
            )

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False
                )
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:  # 更新vision feature和text feature
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:  # 更新text feature
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process 更新 vision feature
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[None, :]
                )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, lower_tokens, higher_tokens, lower_mask=None, V_mask=None):
        Q = lower_tokens.transpose(0, 1)
        K = higher_tokens.transpose(0, 1)
        V = higher_tokens.transpose(0, 1)
        attn_output, _ = self.cross_attention(Q, K, V, key_padding_mask=V_mask)
        # Add & norm
        updated_lower_tokens = self.norm(attn_output.transpose(0, 1) + lower_tokens)
        return updated_lower_tokens


def split_tokens(topk_proposals):
    sorted = torch.sort(topk_proposals, dim=1, descending=False)[0]
    num_lower = int(0.9 * topk_proposals.size(1))
    lower_idxes = sorted[:, :num_lower]
    higher_idxes = sorted[:, num_lower:]

    return lower_idxes, higher_idxes
