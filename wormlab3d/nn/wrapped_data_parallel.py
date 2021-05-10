from itertools import chain

from torch import nn
from torch.nn import DataParallel


class WrappedDataParallel(DataParallel):
    """
    Wrapper class for pytorch's DataParallel class. When a nn.Module is wrapped with DataParallel the attributes
    of the original module are no longer accessible. This class provides a workaround to that problem.
    """

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            try:
                return self.module.__getattr__(item)
            except Exception:
                if hasattr(self.module, item):
                    return getattr(self.module, item)
            raise

    def forward(self, *inputs, **kwargs):
        """
        Override the parent method to gather buffers together on the output device.
        """
        batch_size = inputs[0].shape[0]

        # --------------- As parent: ---------------
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # for forward function without any inputs, empty list and dict will be created
        # so the module can be executed on one device which is the first one in device_ids
        if not inputs and not kwargs:
            inputs = ((),)
            kwargs = ({},)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)

        # ---------------------------------------------

        # Recursively gather the buffers from the different replicas onto the same device
        # and into the module object referenced by this class.
        def gather_buffers(master, module_replicas):
            for k, buff in master._buffers.items():
                buffer_values = [replica._buffers[k] for replica in module_replicas]

                # Check the number of buffers add up to the batch dimension, otherwise ignore
                n_values = sum([v.shape[0] for v in buffer_values if v is not None and v.dim() > 0])
                if n_values == batch_size:
                    master._buffers[k] = self.gather(buffer_values, self.output_device)

            for mk, child_mod in master._modules.items():
                if isinstance(child_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    continue
                child_replicas = [replica._modules[mk] for replica in module_replicas]
                gather_buffers(master._modules[mk], child_replicas)

        gather_buffers(self.module, replicas)

        return self.gather(outputs, self.output_device)
