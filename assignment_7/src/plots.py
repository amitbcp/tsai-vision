import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
import torch
import numpy as np

class Plots:
    def __init__(self):
        pass

    def sampleVisual(dataset):
        batch = next(iter(dataset))
        images, labels = batch
        batch_grid = make_grid(images)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        #return plt.imshow(batch_grid[0].squeeze(), cmap='gray_r')
        return plt.imshow(batch_grid[0].squeeze())
        
    def plotting(model, loader, device):
        wrong_images = []
        wrong_label = []
        correct_label = []

        with torch.no_grad():
            for img, label in loader:
                img, label = img.to(device), label.to(device)
                pred_label = model(img.to(device))
                pred = pred_label.argmax(dim=1, keepdim=True)

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])

                wrong_predictions = list(
                    zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
                fig = plt.figure(figsize=(8, 10))
                fig.tight_layout()
                for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
                    img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
                    ax = fig.add_subplot(5, 2, i+1)
                    ax.axis('off')
                    ax.set_title(
                        f'\nactual {target.item()}\npredicted {pred.item()}', fontsize=10)
                    ax.imshow(img.squeeze(), cmap='gray_r')

                plt.show()
            return len(wrong_predictions)
            
    def stat_graph(train_acc, train_losses, test_acc, test_losses):
      fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
      ax1 = ax[0,0]
      ax1.set_title("TrainAccuracy")
      ax1.plot(train_acc, color="blue")
      ax2 = ax[0,1]
      ax2.set_title("TestAccuracy")
      ax2.plot(test_acc, color="blue")
      ax3 = ax[1,0]
      ax3.set_title("TrainLoss")
      ax3.plot(train_losses, color="blue")
      ax4 = ax[1,1]
      ax4.set_title("TestLoss")
      ax4.plot(test_losses, color="blue")
      plt.show()

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def misclassifications(model,test_loader,device):
      wrong_images=[]
      wrong_label=[]
      correct_label=[]
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)        
              #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              _, predicted = torch.max(output.data, 1)
              # c = (predicted == labels).squeeze()
              # for i in range(4):
              #   label = labels[i]
              #   class_correct[label] += c[i].item()
              #   class_total[label] += 1
                
              wrong_pred = (predicted.eq(target.view_as(predicted)) == False)
              wrong_images.append(data[wrong_pred])
              wrong_label.append(predicted[wrong_pred])
              correct_label.append(target.view_as(predicted)[wrong_pred])  

              wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
          print(f'Total wrong predictions are {len(wrong_predictions)}')
          
          
          fig = plt.figure(figsize=(8,10))
          fig.tight_layout()
          for i, (img, predicted, correct) in enumerate(wrong_predictions[:10]):
            Plots.imshow(torchvision.utils.make_grid(img))
              # img, predicted, target = img.cpu().numpy(), predicted.cpu(), correct.cpu()
              # img = torch.from_numpy(img)
              # ax = fig.add_subplot(5, 2, i+1)
              # ax.axis('off')
              # ax.set_title(f'\nactual {target.item()}\npredicted {predicted.item()}',fontsize=10)  
              # #plt.imshow(np.transpose(npimg, (1, 2, 0)))
              # #ax.imshow(img.squeeze(), cmap='gray_r')  
              # img = img / 2 + 0.5     # unnormalize
              # npimg = img.numpy()
              # plt.imshow(np.transpose(npimg, (1, 2, 0)))
          
              
          plt.show()

    def miscImages(model,  test_loader, device):
      model.eval()
      test_loss = 0
      incorrect = 0
      with torch.no_grad():
          count = 10
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

              for i in range(len(target)):
                if pred[i].item() != target[i]:
                  incorrect += 1
                  print('\n\n{} [ Predicted Value: {}, Actual Value: {} ]'.format(
                  incorrect, pred[i].item(), target[i], ))
                  plt.imshow(data[i].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray_r')
                  plt.show()
                  count = count -1
                if count == 0:
                  break  
              if count == 0:
                break  



