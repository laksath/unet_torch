import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_history(train_history, valid_history, plot_path):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(train_history["train_loss"], label="train_loss")
	plt.plot(valid_history["valid_loss"], label="valid_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")

	plt.savefig(plot_path)
 
def prepare_plot(origImage, origMask, predMask):
	
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
 
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
 
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
 
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()


def make_predictions(model, X, y, threshold=0.5, device='cpu', base = False):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
	
		orig = X.copy()
		gtMask = y.copy()
  
  
  		# make the channel axis to be the leading one, add a batch dimension, create a PyTorch tensor, and flash it to the current device
		image = np.transpose(X, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(device)
  
		# make the prediction, and convert the result to a NumPy array
		if base:
			predMask = model(image).squeeze()
		else:
			predMask = model(image)[0].squeeze()
		predMask = predMask.cpu().numpy()
  
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > threshold) * 255
		predMask = predMask.astype(np.uint8)

		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)


