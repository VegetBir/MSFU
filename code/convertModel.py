import torch
import torch.nn
import onnx
from denoisingUnet import device,denoising_unet


model = denoising_unet()
model.load_state_dict(torch.load('./msd/denoising_unet_p_epoch20.pth'))
model.eval()
model = model.to(device)


input_names = ['input']
output_names = ['output']

x = torch.randn(1, 1, 256, 256, requires_grad=True).to(device)

try:
    torch.onnx.export(model, x, 'msfu.onnx', input_names=input_names, output_names=output_names, verbose=False)
except Exception as e:
    print("An error occurred during ONNX export:", e)