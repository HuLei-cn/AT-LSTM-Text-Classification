import pandas as pd
import matplotlib.pyplot as plt

# 创建文件
# df = pd.DataFrame(columns=['epoch', 'step', 'train_loss', 'test_loss'])
# df.to_csv('train_acc_.csv', index=False)

# data = pd.read_csv('train_acc_42B_pre_emb.csv')
# epoch = 1
# x = data.loc[data['epoch'] == epoch, 'step']
# y1 = data.loc[data['epoch'] == epoch, 'train_loss']
# plt.plot(x, y1, 'g-', label=u'epoch_1')
#
# y3 = data.loc[data['epoch'] == 3, 'train_loss']
# plt.plot(x, y3, 'b-', label=u'epoch_3')
#
# y6 = data.loc[data['epoch'] == 6, 'train_loss']
# plt.plot(x, y6, 'y-', label=u'epoch_6')
#
# y8 = data.loc[data['epoch'] == 8, 'train_loss']
# plt.plot(x, y8, 'r-', label=u'epoch_8')
#
# plt.title(u'train_loss')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('loss')
# plt.show()

# 画两个图：loss和acc分开
# 可视化train_loss
data = pd.read_csv('one_way_lstm.csv')
x = range(0, 1950)
y = data['train_loss']
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', label=u'train_loss')
plt.ylim([0, 1])
plt.title(u'train_loss and test_acc')
plt.legend()
plt.ylabel('loss')

# 可视化测试集准确率
test_data = pd.read_csv('test_acc_one_way_epoch8.csv')
x = range(0, 50)
y = test_data['test_acc']
plt.subplot(2, 1, 2)
plt.plot(x, y, 'r-', label=u'test_acc')
plt.legend()
plt.xlabel('iterations & epochs')
plt.ylabel('acc')
plt.ylim([0, 1])

plt.show()


