import torchvision, torch, random
from torchvision import transforms
import matplotlib.pyplot as plt

clean_tf = transforms.ToTensor();
noise = lambda x,s: torch.clamp(x + s*torch.randn_like(x),0,1);

ds = torchvision.datasets.CIFAR10('data',train=False,download=True,transform=clean_tf);
idxs = random.sample(range(len(ds)),3);
sigmas = [0.0,0.1,0.2,0.3];

# fuzzy!
for k in idxs:
    img,_ = ds[k];
    grid = torch.stack([noise(img,s) for s in sigmas]);
    torchvision.utils.save_image(grid, f'example_{k}.png', nrow=len(sigmas));
