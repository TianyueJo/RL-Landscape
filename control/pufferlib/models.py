from pdb import set_trace as T
import numpy as np
import math

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            num_obs = np.prod(env.single_observation_space.shape)
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )
            
        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_atns), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values

class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        #self.pre_layernorm = nn.LayerNorm(hidden_size)
        #self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        hidden = self.policy.encode_observations(observations, state=state)
        h = state['lstm_h']
        c = state['lstm_c']

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
            lstm_state = (h, c)
        else:
            lstm_state = None

        #hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        #hidden = self.post_layernorm(hidden)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x = observations
        lstm_h = state['lstm_h']
        lstm_c = state['lstm_c']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, 'LSTM state must be (h, c)'
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B*TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        #hidden = self.pre_layernorm(hidden)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()
 
        #hidden = self.post_layernorm(hidden)
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B*TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        #state.batch_logits = logits.reshape(B, TT, -1)
        state['hidden'] = hidden
        state['lstm_h'] = lstm_h.detach()
        state['lstm_c'] = lstm_c.detach()
        return logits, values


class SB3LikeLSTM(nn.Module):
    """Recurrent policy mimicking SB3's MlpLstmPolicy defaults."""

    def __init__(self, env, lstm_hidden_size=64, head_hidden_size=None,
                 activation_fn=nn.ReLU, enable_critic_lstm=True):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.head_hidden_size = head_hidden_size or lstm_hidden_size
        self.enable_critic_lstm = enable_critic_lstm

        self.is_multidiscrete = isinstance(env.single_action_space, pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict)
        except Exception:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict)

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            self.obs_dim = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
        else:
            self.obs_dim = int(np.prod(env.single_observation_space.shape))

        self.actor_lstm = nn.LSTM(self.obs_dim, lstm_hidden_size, num_layers=1)
        self.actor_cell = nn.LSTMCell(self.obs_dim, lstm_hidden_size)
        if enable_critic_lstm:
            self.critic_lstm = nn.LSTM(self.obs_dim, lstm_hidden_size, num_layers=1)
            self.critic_cell = nn.LSTMCell(self.obs_dim, lstm_hidden_size)
        else:
            self.critic_lstm = self.actor_lstm
            self.critic_cell = self.actor_cell

        self.pi_head = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(lstm_hidden_size, self.head_hidden_size)),
            activation_fn(),
        )
        self.vf_head = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(lstm_hidden_size, self.head_hidden_size)),
            activation_fn(),
        )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.policy_head = pufferlib.pytorch.layer_init(nn.Linear(self.head_hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.policy_head = pufferlib.pytorch.layer_init(nn.Linear(self.head_hidden_size, num_atns), std=0.01)
        else:
            action_dim = env.single_action_space.shape[0]
            self.policy_mean = pufferlib.pytorch.layer_init(nn.Linear(self.head_hidden_size, action_dim), std=0.01)
            init_log_std = math.log(0.6)
            self.policy_logstd = nn.Parameter(torch.full((1, action_dim), init_log_std))

        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.head_hidden_size, 1), std=1)
        self.obs_shape = env.single_observation_space.shape

    def _flatten_observations(self, observations):
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else:
            observations = observations.view(batch_size, -1)
        return observations.float()

    def _decode_policy(self, hidden):
        if self.is_multidiscrete:
            logits = self.policy_head(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.policy_mean(hidden)
            logstd = self.policy_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.policy_head(hidden)
        return logits

    def _prepare_cell_state(self, tensor, batch):
        if tensor is None:
            return None
        if tensor.ndim == 3:
            return tensor[-1]
        if tensor.ndim == 2:
            return tensor
        raise ValueError('Invalid LSTM state shape', tensor.shape)

    def _cell_forward(self, cell, inputs, h, c):
        h = self._prepare_cell_state(h, inputs.shape[0])
        c = self._prepare_cell_state(c, inputs.shape[0])
        if h is None or c is None:
            h = inputs.new_zeros(inputs.shape[0], self.hidden_size)
            c = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        h_new, c_new = cell(inputs, (h, c))
        return h_new, (h_new, c_new)

    def forward_eval(self, observations, state):
        flat = self._flatten_observations(observations)

        pi_h, pi_c = state.get('pi_h'), state.get('pi_c')
        vf_h, vf_c = state.get('vf_h'), state.get('vf_c')

        pi_out, pi_cell = self._cell_forward(self.actor_cell, flat, pi_h, pi_c)
        if self.enable_critic_lstm:
            vf_out, vf_cell = self._cell_forward(self.critic_cell, flat, vf_h, vf_c)
        else:
            vf_out, vf_cell = pi_out, pi_cell

        pi_latent = self.pi_head(pi_out)
        vf_latent = self.vf_head(vf_out)
        logits = self._decode_policy(pi_latent)
        values = self.value_head(vf_latent)

        state['pi_h'], state['pi_c'] = pi_cell[0].unsqueeze(0), pi_cell[1].unsqueeze(0)
        if self.enable_critic_lstm:
            state['vf_h'], state['vf_c'] = vf_cell[0].unsqueeze(0), vf_cell[1].unsqueeze(0)
        else:
            state['vf_h'], state['vf_c'] = state['pi_h'], state['pi_c']
        return logits, values

    def forward(self, observations, state):
        x_shape, space_shape = observations.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', observations.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', observations.shape)

        x = observations.reshape(B * TT, *space_shape)
        flat = self._flatten_observations(x)
        flat = flat.reshape(B, TT, self.obs_dim).transpose(0, 1)  # TT, B, obs

        pi_h = state.get('pi_h')
        pi_c = state.get('pi_c')
        if pi_h is not None and pi_c is not None:
            pi_state = (pi_h, pi_c)
        else:
            pi_state = None

        vf_h = state.get('vf_h')
        vf_c = state.get('vf_c')
        if self.enable_critic_lstm:
            if vf_h is not None and vf_c is not None:
                vf_state = (vf_h, vf_c)
            else:
                vf_state = None
        else:
            vf_state = pi_state

        pi_out, (pi_h, pi_c) = self.actor_lstm(flat, pi_state)
        if self.enable_critic_lstm:
            vf_out, (vf_h, vf_c) = self.critic_lstm(flat, vf_state)
        else:
            vf_out, (vf_h, vf_c) = pi_out, (pi_h, pi_c)

        pi_flat = pi_out.transpose(0, 1).reshape(B * TT, self.hidden_size)
        vf_flat = vf_out.transpose(0, 1).reshape(B * TT, self.hidden_size)

        pi_latent = self.pi_head(pi_flat)
        vf_latent = self.vf_head(vf_flat)
        logits = self._decode_policy(pi_latent)
        values = self.value_head(vf_latent).reshape(B, TT)

        state['pi_h'] = pi_h.detach()
        state['pi_c'] = pi_c.detach()
        state['vf_h'] = vf_h.detach()
        state['vf_c'] = vf_c.detach()
        return logits, values

class Convolutional(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        #TODO: Remove these from required params
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class ProcgenResnet(nn.Module):
    '''Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper.'''
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden
 
    def decode_actions(self, hidden):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
