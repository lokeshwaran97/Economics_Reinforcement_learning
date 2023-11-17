from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  


from ray.rllib.models.torch.misc import (
    normc_initializer as torch_normc_initializer,
    SlimFC,
)
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.util import log_once


torch, nn = try_import_torch()

#This class have added the below two class functionalities such as Action masking and processing complex input
#https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
#https://github.com/ray-project/ray/blob/aecc4c8d28c6fa1ac73c1d142fe5b9ee355410a4/rllib/models/torch/complex_input_net.py#L19
class ComplexInputNetwork(TorchModelV2, nn.Module):
    

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if log_once("complex_input_net_deprecation_torch"):
            deprecation_warning(
                old="ray.rllib.models.torch.complex_input_net.ComplexInputNetwork",
            )
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, self.original_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        self.cnns = nn.ModuleDict()
        self.one_hot = nn.ModuleDict()
        self.flatten_dims = {}
        self.flatten = nn.ModuleDict()
        concat_size = 0

        #observation space inputs were iterated to fit the model 
        for i, component in enumerate(self.flattened_input_space):
            i = str(i)
            
            if len(component.shape) == 3 and isinstance(component, Box):
                config = {          #the CNN filter and stride configuration passed during PPO training were assigned here 
                    "conv_filters": model_config["custom_model_config"]["conv_filters"]
                    if "conv_filters" in model_config
                    else get_filter_config(component.shape),
                    "conv_activation": "relu", #relu activation function is used for CNN model
                    "post_fcnet_hiddens": [],
                }
                #CNN model is created 
                self.cnns[i] = ModelCatalog.get_model_v2(
                    component, 
                    action_space,  
                    num_outputs=None,
                    model_config=config, #CNN model is defined based on passed configuration
                    framework="torch",
                    name="cnn_{}".format(i),
                )
                #the number of flattened layer cells were added to concat_size 
                concat_size += self.cnns[i].num_outputs
                self.add_module("cnn_{}".format(i), self.cnns[i])
            elif isinstance(component, (Discrete, MultiDiscrete)):
                if isinstance(component, Discrete):
                    size = component.n
                else:
                    size = np.sum(component.nvec)
                config = {
                    "fcnet_hiddens": [256, 256], # non-spatial model hidden neural network configuration
                    "fcnet_activation": "tanh", # tanh activation function used for spatial model
                    "post_fcnet_hiddens": [],
                }
                self.one_hot[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None,
                    model_config=config, # non-spatial model is created
                    framework="torch",
                    name="one_hot_{}".format(i),
                )
                concat_size += self.one_hot[i].num_outputs
                self.add_module("one_hot_{}".format(i), self.one_hot[i])
            else:
                size = int(np.product(component.shape))
                config = {
                    "fcnet_hiddens": [256, 256], # observation other than Discrete or multidiscrete inputs hidden neural network configuration
                    "fcnet_activation": "tanh",  # were processed in this section
                    "post_fcnet_hiddens": [],
                }
                self.flatten[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32), # this will not be triggered for this project 
                    action_space,   
                    num_outputs=None,
                    model_config=config, 
                    framework="torch",
                    name="flatten_{}".format(i),
                )
                self.flatten_dims[i] = size
                concat_size += self.flatten[i].num_outputs
                self.add_module("flatten_{}".format(i), self.flatten[i])

        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", [256,256]), # after spatial and non-spatial model were connected and flattened
            "fcnet_activation": model_config.get("post_fcnet_activation", "relu"), # this configurations used to create two hidden networks with relu activation function

        }
        # all the model outputs are flattened and were connected to two hidden neural network output from this layer will be connected to LSTM model
        #post FC network
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack",
        )

        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        if num_outputs:
            #output form the post FC network will be connected to actor model
            #seperate model will be created for actor model
            self.logits_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=num_outputs, #agent action probabilites is the output
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
            #ouput from the post FC network will be connected to critic model
            #seperate model will be created for critic model
            self.value_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=1, #output of critic model is state value
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
        else:
            self.num_outputs = concat_size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        
        
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )

        outs = []
        for i, component in enumerate(tree.flatten(orig_obs)):
            i = str(i)
            if i in self.cnns:
                cnn_out, _ = self.cnns[i](SampleBatch({SampleBatch.OBS: component}))
                outs.append(cnn_out)
            elif i in self.one_hot:
                if component.dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ]:
                    one_hot_in = {
                        SampleBatch.OBS: one_hot(
                            component, self.flattened_input_space[int(i)]
                        )
                    }
                else:
                    one_hot_in = {SampleBatch.OBS: component}
                one_hot_out, _ = self.one_hot[i](SampleBatch(one_hot_in))
                outs.append(one_hot_out)
            else:
                nn_out, _ = self.flatten[i](
                    SampleBatch(
                        {
                            SampleBatch.OBS: torch.reshape(
                                component, [-1, self.flatten_dims[i]]
                            )
                        }
                    )
                )
                outs.append(nn_out)

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        #agent action probabilites and state value ouput is got from actor and critic model
        logits, values = self.logits_layer(out), self.value_layer(out)
        #agent mask of the agent
        action_mask = orig_obs["action_mask"]
        huge_negative_val = -1000000
        negative_values = torch.ones_like(action_mask) * huge_negative_val
        #large negative values is assigned for invalid action
        mask = negative_values * (1 - action_mask)
        #action mask is added to agent action probabilities
        masked_logits = logits + mask
        self._value_out = torch.reshape(values, [-1])
        

        return masked_logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out