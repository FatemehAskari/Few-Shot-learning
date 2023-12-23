from torchvision import transforms
from torch import nn, optim
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names


class PrototypicalNetworks(nn.Module):
    def __init__(self,backbone:nn.Module,weightlearnable):
        super(PrototypicalNetworks, self).__init__()
        self.backbone=backbone
        self.weightlearnable=weightlearnable
        
        self.weight_1 = nn.Parameter(torch.Tensor([1]))
        self.weight_2 = nn.Parameter(torch.Tensor([2]))
        self.weight_3 = nn.Parameter(torch.Tensor([3]))
        self.weight_4 = nn.Parameter(torch.Tensor([4]))
        self.weight_5 = nn.Parameter(torch.Tensor([5]))

    def calculate(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor,num,numker):
        global_avg_pool = nn.AdaptiveAvgPool2d((1))
        train_nodes, eval_nodes = get_graph_node_names(self.backbone)
        return_nodes2 = {
            str(train_nodes[num]): "maxpool1"
        }
        f2 = create_feature_extractor(self.backbone, return_nodes=return_nodes2)
        out2 = f2(support_images)
        out2_pooled = global_avg_pool(out2['maxpool1'])
        z_support  = out2_pooled.view(out2_pooled.size(0), -1)

        out3 = f2(query_images)
        out3_pooled = global_avg_pool(out3['maxpool1'])
        z_query  = out3_pooled.view(out3_pooled.size(0), -1)
        n_way=len(torch.unique(support_labels))

        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        dists = torch.cdist(z_query, z_proto)
        return dists
    def forward(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor):
        d1=self.calculate(support_images,support_labels,query_images,4,64)
        d2=self.calculate(support_images,support_labels,query_images,19,128)
        d3=self.calculate(support_images,support_labels,query_images,35,256)
        d4=self.calculate(support_images,support_labels,query_images,51,512)
        d5=self.calculate(support_images,support_labels,query_images,67,512)
        if self.weightlearnable:
            scores4 = -(self.weight_1 * d1 + self.weight_2 * d2 + self.weight_3 * d3 + self.weight_4 * d4+self.weight_5 * d5)
        else:
           scores4 = -(d1+d2+d3+d4+d5)     
        return scores4