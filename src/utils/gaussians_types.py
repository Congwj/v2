from torch import Tensor


class Gaussians:
    def __init__(
        self,
        means=None,
        covariances=None,
        harmonics=None,
        opacities=None,
        scales=None,
        rotations=None,
        semantic_features=None,
        semantic_labels=None,
        instance_labels=None,
        features=None,
        seg_query_class_logits=None,
        **kwargs,
    ):
        self.means: Tensor = means
        self.covariances: Tensor = covariances
        self.harmonics: Tensor = harmonics
        self.opacities: Tensor = opacities
        self.scales: Tensor = scales
        self.rotations: Tensor = rotations
        self.semantic_features: Tensor = semantic_features
        self.semantic_labels: Tensor = semantic_labels
        self.instance_labels: Tensor = instance_labels
        self.features: Tensor = features
        self.seg_query_class_logits = seg_query_class_logits
        for key, value in kwargs.items():
            setattr(self, key, value)

    def detach_cpu_copy(self):
        copy_gaussians = Gaussians()
        for field_name, field_value in vars(self).items():
            if isinstance(field_value, Tensor):
                setattr(copy_gaussians, field_name, field_value.detach().cpu())
            else:
                setattr(copy_gaussians, field_name, field_value)
        return copy_gaussians
