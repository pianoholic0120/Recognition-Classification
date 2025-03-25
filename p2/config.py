exp_name = 'default'  # name of experiment

# Model Options
# model_type = 'resnet18'  # 'mynet' or 'resnet18'
model_type = 'mynet' 

# Learning Options
epochs = 30                # train how many epochs
batch_size = 32            # batch size for dataloader 
use_adam = False         # Adam or SGD optimizer
lr = 1e-2                  # learning rate 1e-2 for SGD, 1e-3 for Adam
milestones = [16, 24]  # reduce learning rate at 'milestones' epochs, [16, 32, 45] for 50 epochs