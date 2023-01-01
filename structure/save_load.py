import torch

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, path, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.path = path
    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print("Best validation dice loss: {:.4f}".format(self.best_valid_loss)+f" , Saving best model for epoch: {epoch+1} at {self.path}.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, self.path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

def load_ckp_cpu(checkpoint_fpath, model, optimizer):
	checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return model, optimizer, checkpoint['epoch']