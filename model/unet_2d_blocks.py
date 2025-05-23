import torch
from torch import nn

from diffusers.models.attention import AttentionBlock
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from model.attention import Transformer2DModel

def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_head_channels=attn_num_head_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        print("\n----------------------- UNetMidBlock2DCrossAttn -----------------------")
        print("in_channels: ", in_channels)
        print("temb_channels: ", temb_channels)
        print("num_layers: ", num_layers)
        print("resnet_eps: ", resnet_eps)
        print("resnet_time_scale_shift: ", resnet_time_scale_shift)
        print("resnet_act_fn: ", resnet_act_fn)
        print("resnet_groups: ", resnet_groups)
        print("resnet_pre_norm: ", resnet_pre_norm)
        print("attn_num_head_channels: ", attn_num_head_channels)
        print("cross_attention_dim: ", cross_attention_dim)
        print("use_linear_projection: ", use_linear_projection)
        print("upcast_attention: ", upcast_attention)
        print("output_scale_factor: ", output_scale_factor)
        print("dropout: ", dropout)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
                
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states, temb=None, image_hidden_states=None, encoder_hidden_states=None, cross_attention_kwargs=None
    ):
        print("\n----------------------- UNetMidBlock2DCrossAttn forward -----------------------")
        print("hidden_states: ", hidden_states.shape)
        print("temb: ", temb.shape if temb is not None else "None")
        print("encoder_hidden_states: ", encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
        if image_hidden_states is not None:
            print("image_hidden_states:")
            for key, value in image_hidden_states.items():
                print(f"{key}: {value.shape}")
        if cross_attention_kwargs is not None:
            print("cross_attention_kwargs:")
            for key, value in cross_attention_kwargs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {value}")

        hidden_states = self.resnets[0](hidden_states, temb)
        mid_img_dif_conditions = []

        if image_hidden_states is None:
            print("No image_hidden_states")
            # No image_hidden_states, ref_image cycle, need img_dif_condition
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                result = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                hidden_states = result.sample
                print("hidden_states.shape after attn: ", hidden_states.shape)
                mid_img_dif_conditions.append(result.img_dif_condition)
                hidden_states = resnet(hidden_states, temb)
                print("hidden_states.shape after resnet: ", hidden_states.shape)
        else:
            print("Have image_hidden_states")
            # Have image_hidden_states, image cycle, no need for img_dif_condition
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                hidden_states = attn(
                    hidden_states,
                    image_hidden_states=image_hidden_states["mid"],
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                print("hidden_states.shape after attn: ", hidden_states.shape)
                hidden_states = resnet(hidden_states, temb)
                print("hidden_states.shape after resnet: ", hidden_states.shape)

        print("hidden_states: ", hidden_states.shape)
        print("mid_img_dif_conditions: ", [x.shape for x in mid_img_dif_conditions])
        return hidden_states, mid_img_dif_conditions


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        print("\n----------------------- CrossAttnDownBlock2D -----------------------")
        print("in_channels: ", in_channels)
        print("out_channels: ", out_channels)
        print("temb_channels: ", temb_channels)
        print("num_layers: ", num_layers)
        print("resnet_eps: ", resnet_eps)
        print("resnet_time_scale_shift: ", resnet_time_scale_shift)
        print("resnet_act_fn: ", resnet_act_fn)
        print("resnet_groups: ", resnet_groups)
        print("resnet_pre_norm: ", resnet_pre_norm)
        print("attn_num_head_channels: ", attn_num_head_channels)
        print("cross_attention_dim: ", cross_attention_dim)
        print("use_linear_projection: ", use_linear_projection)
        print("only_cross_attention: ", only_cross_attention)
        print("upcast_attention: ", upcast_attention)
        print("output_scale_factor: ", output_scale_factor)
        print("downsample_padding: ", downsample_padding)
        print("add_downsample: ", add_downsample)
        print("dropout: ", dropout)

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, temb=None, image_hidden_states=None, encoder_hidden_states=None, cross_attention_kwargs=None
    ):
        output_states = ()
        down_img_dif_conditions = []
        
        ln = 4 - hidden_states.shape[2] // 8
        if ln <1: ln=1 # the number of the block: 1 or 2 or 3

        print("\n----------------------- CrossAttnDownBlock2D forward -----------------------")
        print("hidden_states: ", hidden_states.shape)
        print("temb: ", temb.shape if temb is not None else "None")
        print("encoder_hidden_states: ", encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
        print("ln: ", ln)
        print("gradient_checkpointing: ", self.gradient_checkpointing)
        if image_hidden_states is not None:
            print("image_hidden_states:")
            for key, value in image_hidden_states.items():
                print(f"{key}: {value.shape}")
        if cross_attention_kwargs is not None:
            print("cross_attention_kwargs:")
            for key, value in cross_attention_kwargs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {value}")

        if image_hidden_states is None:
            # No image_hidden_states, ref_image cycle, need img_dif_condition
            print("No image_hidden_states")
            for resnet, attn in zip(self.resnets, self.attentions):
                hidden_states = resnet(hidden_states, temb)
                print("hidden_states.shape after resnet: ", hidden_states.shape)
                result = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                hidden_states = result.sample
                print("hidden_states.shape after attn: ", hidden_states.shape)
                down_img_dif_conditions.append(result.img_dif_condition)

                output_states += (hidden_states,)
        else:
            # Have image_hidden_states, image cycle, no need for img_dif_condition
            print("Have image_hidden_states")
            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        image_hidden_states["down_"+str(ln)+'_'+str(i+1)],
                        encoder_hidden_states,
                        cross_attention_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    print("hidden_states.shape after resnet: ", hidden_states.shape)
                    hidden_states = attn(
                        hidden_states,
                        image_hidden_states=image_hidden_states["down_"+str(ln)+'_'+str(i+1)],
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    print("hidden_states.shape after attn: ", hidden_states.shape)

                output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                print("hidden_states.shape after downsampler: ", hidden_states.shape)

            output_states += (hidden_states,)

        print("hidden_states: ", hidden_states.shape)
        print("output_states: ", [x.shape for x in output_states])
        print("down_img_dif_conditions: ", [x.shape for x in down_img_dif_conditions])
        return hidden_states, output_states, down_img_dif_conditions


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        print("\n----------------------- DownBlock2D -----------------------")
        print("in_channels: ", in_channels)
        print("out_channels: ", out_channels)
        print("temb_channels: ", temb_channels)
        print("num_layers: ", num_layers)
        print("resnet_eps: ", resnet_eps)
        print("resnet_time_scale_shift: ", resnet_time_scale_shift)
        print("resnet_act_fn: ", resnet_act_fn)
        print("resnet_groups: ", resnet_groups)
        print("resnet_pre_norm: ", resnet_pre_norm)
        print("output_scale_factor: ", output_scale_factor)
        print("add_downsample: ", add_downsample)
        print("downsample_padding: ", downsample_padding)
        print("dropout: ", dropout)

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        print("\n----------------------- DownBlock2D forward -----------------------")
        print("hidden_states: ", hidden_states.shape)
        print("temb: ", temb.shape if temb is not None else "None")
        print("gradient_checkpointing: ", self.gradient_checkpointing)

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
                print("hidden_states.shape after resnet: ", hidden_states.shape)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                print("hidden_states.shape after downsampler: ", hidden_states.shape)

            output_states += (hidden_states,)
        
        print("hidden_states: ", hidden_states.shape)
        print("output_states: ", [x.shape for x in output_states])
        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        print("\n----------------------- CrossAttnUpBlock2D -----------------------")
        print("in_channels: ", in_channels)
        print("out_channels: ", out_channels)
        print("prev_output_channel: ", prev_output_channel)
        print("temb_channels: ", temb_channels)
        print("num_layers: ", num_layers)
        print("resnet_eps: ", resnet_eps)
        print("resnet_time_scale_shift: ", resnet_time_scale_shift)
        print("resnet_act_fn: ", resnet_act_fn)
        print("resnet_groups: ", resnet_groups)
        print("resnet_pre_norm: ", resnet_pre_norm)
        print("attn_num_head_channels: ", attn_num_head_channels)
        print("cross_attention_dim: ", cross_attention_dim)
        print("use_linear_projection: ", use_linear_projection)
        print("only_cross_attention: ", only_cross_attention)
        print("upcast_attention: ", upcast_attention)
        print("output_scale_factor: ", output_scale_factor)
        print("add_upsample: ", add_upsample)
        print("dropout: ", dropout)

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        image_hidden_states=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
    ):
        up_img_dif_conditions = []

        ln = hidden_states.shape[2] // 8
        if ln > 3: ln=3 # the number of the block: 1 or 2 or 3

        print("\n----------------------- CrossAttnUpBlock2D forward -----------------------")
        print("hidden_states: ", hidden_states.shape)
        print("res_hidden_states_tuple: ", [x.shape for x in res_hidden_states_tuple])
        print("temb: ", temb.shape if temb is not None else "None")
        print("encoder_hidden_states: ", encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
        print("upsample_size: ", upsample_size)
        print("ln: ", ln)
        print("gradient_checkpointing: ", self.gradient_checkpointing)
        if image_hidden_states is not None:
            print("image_hidden_states:")
            for key, value in image_hidden_states.items():
                print(f"{key}: {value.shape}")
        if cross_attention_kwargs is not None:
            print("cross_attention_kwargs:")
            for key, value in cross_attention_kwargs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {value}")

        if image_hidden_states is None:
            # No image_hidden_states, ref_image cycle, need img_dif_condition
            print("No image_hidden_states")
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                print("hidden_states.shape after concat: ", hidden_states.shape)

                hidden_states = resnet(hidden_states, temb)
                print("hidden_states.shape after resnet: ", hidden_states.shape)

                result = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                hidden_states = result.sample
                print("hidden_states.shape after attn: ", hidden_states.shape)
                up_img_dif_conditions.append(result.img_dif_condition)

        else:
            # Have image_hidden_states, image cycle, no need for img_dif_condition
            print("Have image_hidden_states")
            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                print("hidden_states.shape after concat: ", hidden_states.shape)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        image_hidden_states["up_"+str(ln)+'_'+str(i+1)],
                        encoder_hidden_states,
                        cross_attention_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    print("hidden_states.shape after resnet: ", hidden_states.shape)
                    hidden_states = attn(
                        hidden_states,
                        image_hidden_states=image_hidden_states["up_"+str(ln)+'_'+str(i+1)],
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    print("hidden_states.shape after attn: ", hidden_states.shape)
                

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
                print("hidden_states.shape after upsampler: ", hidden_states.shape)

        print("hidden_states: ", hidden_states.shape)
        print("up_img_dif_conditions: ", [x.shape for x in up_img_dif_conditions])
        return hidden_states, up_img_dif_conditions


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        print("\n----------------------- UpBlock2D -----------------------")
        print("in_channels: ", in_channels)
        print("prev_output_channel: ", prev_output_channel)
        print("out_channels: ", out_channels)
        print("temb_channels: ", temb_channels)
        print("num_layers: ", num_layers)
        print("resnet_eps: ", resnet_eps)
        print("resnet_time_scale_shift: ", resnet_time_scale_shift)
        print("resnet_act_fn: ", resnet_act_fn)
        print("resnet_groups: ", resnet_groups)
        print("resnet_pre_norm: ", resnet_pre_norm)
        print("output_scale_factor: ", output_scale_factor)
        print("add_upsample: ", add_upsample)
        print("dropout: ", dropout)

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        print("\n----------------------- UpBlock2D forward -----------------------")
        print("hidden_states: ", hidden_states.shape)
        print("temb: ", temb.shape if temb is not None else "None")
        print("res_hidden_states_tuple: ", [x.shape for x in res_hidden_states_tuple])
        print("upsample_size: ", upsample_size)
        print("gradient_checkpointing: ", self.gradient_checkpointing)

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            print("hidden_states.shape after concat: ", hidden_states.shape)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
                print("hidden_states.shape after resnet: ", hidden_states.shape)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
                print("hidden_states.shape after upsampler: ", hidden_states.shape)

        print("hidden_states: ", hidden_states.shape)
        return hidden_states