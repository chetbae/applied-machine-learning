import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20, 25, 30]
# 100%
category_train100 = [2.288, 1.174, 0.572, 0.423, 0.343, 0.294, 0.264]
category_val100 = [2.053, 1.218, 0.685, 0.599, 0.706, 0.495, 0.497]
category_acc100 = [0.205, 0.581, 0.769, 0.806, 0.782, 0.845, 0.843]


print('100% of training data')
# category
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, category_train100, "-b", label="Training")
ax1.plot(x, category_val100, "-r", label="Validation")
ax1.legend(loc="upper right")
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylim(0)
ax1.set_title('Category')
ax1.set_aspect(1)

ax2.plot(x, category_acc100, "-g", label="Accuracy")
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 1)
ax2.set_title('Category')
ax2.set_aspect(1)

plt.show()

# speech