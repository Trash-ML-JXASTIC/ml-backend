import tensorflow as tf
import linecache
import cv2
import numpy as np
import os
from pathlib import Path

from select_object import pretreatment_image

train_data_root = Path("./data/train")
for item in train_data_root.iterdir():
    print("Train item:", item)

validation_data_root = Path("./data/validation")
for item in validation_data_root.iterdir():
    print("Validation item:", item)

all_train_image_paths = list(train_data_root.glob("*/*"))
all_train_image_paths = [str(path) for path in all_train_image_paths]

all_validation_image_paths = list(validation_data_root.glob("*/*"))
all_validation_image_paths = [str(path) for path in all_validation_image_paths]

train_image_count = len(all_train_image_paths)
print("Train image count:", train_image_count)

validation_image_count = len(all_validation_image_paths)
print("Validation image count:", validation_image_count)

label_names = sorted(
    item.name for item in train_data_root.glob("*/") if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_train_image_labels = [label_to_index[Path(path).parent.name] for path in all_train_image_paths]
all_validation_image_labels = [label_to_index[Path(path).parent.name] for path in all_validation_image_paths]

print("Label names:", label_names)
print("Label to index:", label_to_index)
print("All train image paths:", all_train_image_paths)
print("All train image labels:", all_train_image_labels)
print("All validation image paths:", all_validation_image_paths)
print("All validation image labels:", all_validation_image_labels)


def load_train_dataset(index):  # 从1开始
    if index > train_image_count:
        if index % train_image_count == 0:
            index = train_image_count
        else:
            index %= train_image_count
    # line_str = linecache.getline(train_labels_path, index)
    # image_name, image_label = line_str.split(' ')
    image_name = all_train_image_paths[index]
    image_label = all_train_image_labels[index]
    image = cv2.imread(image_name) # train_images_path + 
    # cv2.imshow('pic',image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    print("--")
    return image, image_label


def combine_train_dataset(count, size):
    train_images_load = np.zeros(shape=(size, 224, 224, 3))
    train_labels_load = np.zeros(shape=(size, len(label_names)))
    for i in range(size):
        train_images_load[i], train_labels_index = load_train_dataset(
            count + i + 1)
        train_labels_load[i][int(train_labels_index) - 1] = 1.0
    count += size
    return train_images_load, train_labels_load, count


def load_test_dataset(index):  # 从1开始
    if index > validation_image_count:
        if index % validation_image_count == 0:
            index = validation_image_count
        else:
            index %= validation_image_count
    # line_str = linecache.getline(test_labels_path, index)
    # image_name, image_label = line_str.split(' ')
    image_name = all_validation_image_paths[index]
    image_label = all_validation_image_labels[index]
    image = cv2.imread(image_name) # test_images_path + 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image, image_label


def combine_test_dataset(count, size):
    test_images_load = np.zeros(shape=(size, 224, 224, 3))
    test_labels_load = np.zeros(shape=(size, len(label_names)))
    for i in range(size):
        test_images_load[i], test_labels_index = load_test_dataset(
            count + i + 1)
        test_labels_load[i][int(test_labels_index) - 1] = 1.0
    count += size
    return test_images_load, test_labels_load, count


# # 通过L2正则化防止过拟合
# def weight_variable_with_loss(shape, stddev, lam):
#     weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
#     if lam is not None:
#         weight_loss = tf.multiply(tf.nn.l2_loss(weight), lam, name='weight_loss')
#         tf.add_to_collection('losses', weight_loss)
#     return weight

def weight_variable(shape, n, use_l2, lam):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=1 / n))
    # L2正则化
    if use_l2 is True:
        weight_loss = tf.multiply(tf.nn.l2_loss(
            weight), lam, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return weight


def bias_variable(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 输入层
with tf.name_scope('input_layer'):
    x_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_input = tf.placeholder(tf.float32, [None, len(label_names)])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    is_use_l2 = tf.placeholder(tf.bool)
    lam = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    # 数据集平均RGB值
    mean = tf.constant([159.780, 139.802, 119.047],
                       dtype=tf.float32, shape=[1, 1, 1, 3])
    x_input = x_input - mean

# 第一个卷积层 size:224
# 卷积核1[3, 3, 3, 64]
# 卷积核2[3, 3, 64, 64]
with tf.name_scope('conv1_layer'):
    w_conv1 = weight_variable([3, 3, 3, 64], 64, use_l2=False, lam=0)
    b_conv1 = bias_variable([64])
    conv_kernel1 = conv2d(x_input, w_conv1)
    bn1 = tf.layers.batch_normalization(conv_kernel1, training=is_training)
    conv1 = tf.nn.relu(tf.nn.bias_add(bn1, b_conv1))

    w_conv2 = weight_variable([3, 3, 64, 64], 64, use_l2=False, lam=0)
    b_conv2 = bias_variable([64])
    conv_kernel2 = conv2d(conv1, w_conv2)
    bn2 = tf.layers.batch_normalization(conv_kernel2, training=is_training)
    conv2 = tf.nn.relu(tf.nn.bias_add(bn2, b_conv2))

    pool1 = max_pool_2x2(conv2)  # 224*224 -> 112*112
    result1 = pool1

# 第二个卷积层 size:112
# 卷积核3[3, 3, 64, 128]
# 卷积核4[3, 3, 128, 128]
with tf.name_scope('conv2_layer'):
    w_conv3 = weight_variable([3, 3, 64, 128], 128, use_l2=False, lam=0)
    b_conv3 = bias_variable([128])
    conv_kernel3 = conv2d(result1, w_conv3)
    bn3 = tf.layers.batch_normalization(conv_kernel3, training=is_training)
    conv3 = tf.nn.relu(tf.nn.bias_add(bn3, b_conv3))

    w_conv4 = weight_variable([3, 3, 128, 128], 128, use_l2=False, lam=0)
    b_conv4 = bias_variable([128])
    conv_kernel4 = conv2d(conv3, w_conv4)
    bn4 = tf.layers.batch_normalization(conv_kernel4, training=is_training)
    conv4 = tf.nn.relu(tf.nn.bias_add(bn4, b_conv4))

    pool2 = max_pool_2x2(conv4)  # 112*112 -> 56*56
    result2 = pool2

# 第三个卷积层 size:56
# 卷积核5[3, 3, 128, 256]
# 卷积核6[3, 3, 256, 256]
# 卷积核7[3, 3, 256, 256]
with tf.name_scope('conv3_layer'):
    w_conv5 = weight_variable([3, 3, 128, 256], 256, use_l2=False, lam=0)
    b_conv5 = bias_variable([256])
    conv_kernel5 = conv2d(result2, w_conv5)
    bn5 = tf.layers.batch_normalization(conv_kernel5, training=is_training)
    conv5 = tf.nn.relu(tf.nn.bias_add(bn5, b_conv5))

    w_conv6 = weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
    b_conv6 = bias_variable([256])
    conv_kernel6 = conv2d(conv5, w_conv6)
    bn6 = tf.layers.batch_normalization(conv_kernel6, training=is_training)
    conv6 = tf.nn.relu(tf.nn.bias_add(bn6, b_conv6))

    w_conv7 = weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
    b_conv7 = bias_variable([256])
    conv_kernel7 = conv2d(conv6, w_conv7)
    bn7 = tf.layers.batch_normalization(conv_kernel7, training=is_training)
    conv7 = tf.nn.relu(tf.nn.bias_add(bn7, b_conv7))

    pool3 = max_pool_2x2(conv7)  # 56*56 -> 28*28
    result3 = pool3

# 第四个卷积层 size:28
# 卷积核8[3, 3, 256, 512]
# 卷积核9[3, 3, 512, 512]
# 卷积核10[3, 3, 512, 512]
with tf.name_scope('conv4_layer'):
    w_conv8 = weight_variable([3, 3, 256, 512], 512, use_l2=False, lam=0)
    b_conv8 = bias_variable([512])
    conv_kernel8 = conv2d(result3, w_conv8)
    bn8 = tf.layers.batch_normalization(conv_kernel8, training=is_training)
    conv8 = tf.nn.relu(tf.nn.bias_add(bn8, b_conv8))

    w_conv9 = weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
    b_conv9 = bias_variable([512])
    conv_kernel9 = conv2d(conv8, w_conv9)
    bn9 = tf.layers.batch_normalization(conv_kernel9, training=is_training)
    conv9 = tf.nn.relu(tf.nn.bias_add(bn9, b_conv9))

    w_conv10 = weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
    b_conv10 = bias_variable([512])
    conv_kernel10 = conv2d(conv9, w_conv10)
    bn10 = tf.layers.batch_normalization(conv_kernel10, training=is_training)
    conv10 = tf.nn.relu(tf.nn.bias_add(bn10, b_conv10))

    pool4 = max_pool_2x2(conv10)  # 28*28 -> 14*14
    result4 = pool4

# 第五个卷积层 size:14
# 卷积核11[3, 3, 512, 512]
# 卷积核12[3, 3, 512, 512]
# 卷积核13[3, 3, 512, 512]
with tf.name_scope('conv5_layer'):
    w_conv11 = weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
    b_conv11 = bias_variable([512])
    conv_kernel11 = conv2d(result4, w_conv11)
    bn11 = tf.layers.batch_normalization(conv_kernel11, training=is_training)
    conv11 = tf.nn.relu(tf.nn.bias_add(bn11, b_conv11))

    w_conv12 = weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
    b_conv12 = bias_variable([512])
    conv_kernel12 = conv2d(conv11, w_conv12)
    bn12 = tf.layers.batch_normalization(conv_kernel12, training=is_training)
    conv12 = tf.nn.relu(tf.nn.bias_add(bn12, b_conv12))

    w_conv13 = weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
    b_conv13 = bias_variable([512])
    conv_kernel13 = conv2d(conv12, w_conv13)
    bn13 = tf.layers.batch_normalization(conv_kernel13, training=is_training)
    conv13 = tf.nn.relu(tf.nn.bias_add(bn13, b_conv13))

    pool5 = max_pool_2x2(conv13)  # 14*14 -> 7*7
    result5 = pool5

# 第一个全连接层 size:7
# 隐藏层节点数 4096
with tf.name_scope('fc1_layer'):
    w_fc14 = weight_variable([7 * 7 * 512, 4096], 4096,
                             use_l2=is_use_l2, lam=lam)
    b_fc14 = bias_variable([4096])
    result5_flat = tf.reshape(result5, [-1, 7 * 7 * 512])
    fc14 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result5_flat, w_fc14), b_fc14))
    # result6 = fc14
    result6 = tf.nn.dropout(fc14, keep_prob)

# 第二个全连接层
# 隐藏层节点数 4096
with tf.name_scope('fc2_layer'):
    w_fc15 = weight_variable([4096, 4096], 4096, use_l2=is_use_l2, lam=lam)
    b_fc15 = bias_variable([4096])
    fc15 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result6, w_fc15), b_fc15))
    # result7 = fc15
    result7 = tf.nn.dropout(fc15, keep_prob)

# 输出层
with tf.name_scope('output_layer'):
    w_fc16 = weight_variable([4096, len(label_names)],
                             len(label_names), use_l2=is_use_l2, lam=lam)
    b_fc16 = bias_variable([len(label_names)])
    fc16 = tf.matmul(result7, w_fc16) + b_fc16
    logits = tf.nn.softmax(fc16)

# 损失函数
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=fc16, labels=y_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', loss)

# 训练函数
with tf.name_scope('train'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 计算准确率
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# 会话初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
save_dir = "models"
checkpoint_name = "train.ckpt"
merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
# writer_train = tf.summary.FileWriter('logs/train', sess.graph)  # 将训练日志写入到logs文件夹下
# writer_test = tf.summary.FileWriter('logs/test', sess.graph)  # 将训练日志写入到logs文件夹下

# 变量初始化
training_steps = 25000
display_step = 10
batch_size = 20
train_images_count = 0
test_images_count = 0
train_avg_accuracy = 0
test_avg_accuracy = 0

# 训练
print("Training start...")

# 模型恢复
# sess = tf.InteractiveSession()
# saver.restore(sess, os.path.join(save_dir, checkpoint_name))
# print("Model restore success！")

for step in range(training_steps):
    train_images, train_labels, train_images_count = combine_train_dataset(
        train_images_count, batch_size)
    test_images, test_labels, test_images_count = combine_test_dataset(
        test_images_count, batch_size)

    # 训练
    if step < 10000:
        train_step.run(
            feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.8, is_training: True, is_use_l2: True,
                       learning_rate: 0.0001, lam: 0.004})
    elif step < 20000:
        train_step.run(
            feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.8, is_training: True, is_use_l2: True,
                       learning_rate: 0.0001, lam: 0.001})
    else:
        train_step.run(
            feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.8, is_training: True, is_use_l2: True,
                       learning_rate: 0.00001, lam: 0.001})

    # 每训练10步，输出显示训练过程
    if step % display_step == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 1.0, is_training: False,
                       is_use_l2: False})
        train_loss = sess.run(loss, feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 1.0,
                                               is_training: False, is_use_l2: False})
        train_result = sess.run(tf.argmax(logits, 1),
                                feed_dict={x_input: train_images, keep_prob: 1.0, is_training: False, is_use_l2: False})
        train_label = sess.run(tf.argmax(y_input, 1), feed_dict={
                               y_input: train_labels})

        test_accuracy = accuracy.eval(
            feed_dict={x_input: test_images, y_input: test_labels, keep_prob: 1.0, is_training: False,
                       is_use_l2: False})
        test_result = sess.run(tf.argmax(logits, 1),
                               feed_dict={x_input: test_images, keep_prob: 1.0, is_training: False, is_use_l2: False})
        test_label = sess.run(tf.argmax(y_input, 1),
                              feed_dict={y_input: test_labels})

        print("Training dataset:")
        print(train_result)
        print(train_label)
        print("Testing dataset:")
        print(test_result)
        print(test_label)

        print("step {}\n training accuracy {}\n loss {}\n testing accuracy {}\n".format(
            step, train_accuracy, train_loss, test_accuracy))
        train_avg_accuracy += train_accuracy
        test_avg_accuracy += test_accuracy
        result_train = sess.run(merged, feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 1.0,
                                                   is_training: False, is_use_l2: False})  # 计算需要写入的日志数据
        writer_train.add_summary(result_train, step)  # 将日志数据写入文件

        result_test = sess.run(merged, feed_dict={x_input: test_images, y_input: test_labels, keep_prob: 1.0,
                                                  is_training: False, is_use_l2: False})  # 计算需要写入的日志数据
        writer_test.add_summary(result_test, step)  # 将日志数据写入文件

    # 每训练100步，显示输出训练平均准确度，保存模型
    if step % (display_step * 10) == 0 and step != 0:
        print("train_avg_accuracy {}".format(train_avg_accuracy / 10))
        train_avg_accuracy = 0
        print("test_avg_accuracy {}".format(test_avg_accuracy / 10))
        test_avg_accuracy = 0

        saver.save(sess, os.path.join(save_dir, checkpoint_name))
        print("Model save success!\n")

print("Training finish...")

# 模型保存
saver.save(sess, os.path.join(save_dir, checkpoint_name))
print("\nModel save success!")

# print("\nTesting start...")
# avg_accuracy = 0
# for i in range(int(validation_image_count / 30) + 1):
#     test_images, test_labels, test_images_count = combine_test_dataset(test_images_count, 30)
#     test_accuracy = accuracy.eval(
#         feed_dict={x_input: test_images, y_input: test_labels, keep_prob: 1.0, is_training: False, is_use_l2: False})
#     test_result = sess.run(tf.argmax(logits, 1),
#                            feed_dict={x_input: test_images, keep_prob: 1.0, is_training: False, is_use_l2: False})
#     test_label = sess.run(tf.argmax(y_input, 1), feed_dict={y_input: test_labels})
#     print(test_result)
#     print(test_label)
#     print("test accuracy {}".format(test_accuracy))
#     avg_accuracy += test_accuracy
#
# print("\ntest_avg_accuracy {}".format(avg_accuracy / (int(validation_image_count / 30) + 1)))

sess.close()


# 识别
# 模型恢复
sess = tf.InteractiveSession()
saver.restore(sess, os.path.join(save_dir, checkpoint_name))
print("Model restore success！")


def predict_img(img_path):
    img = cv2.imread(img_path)
    image = np.reshape(img, [1, 224, 224, 3])
    classify_result = sess.run(tf.argmax(logits, 1),
                               feed_dict={x_input: image, keep_prob: 1.0, is_training: False, is_use_l2: False})
    probability = sess.run(logits, feed_dict={x_input: image, keep_prob: 1.0, is_training: False,
                                              is_use_l2: False}).flatten().tolist()[
        classify_result[0]]
    return classify_result[0], probability


def trash_classify(img_path, img_name, upload_path):
    img_name = img_name.rsplit('.', 1)[0]
    # print(img_name)
    pretrian_img_path, selected_img_path = pretreatment_image(
        img_path, img_name, upload_path)
    predict_result, predict_probability = predict_img(pretrian_img_path)
    return predict_result, predict_probability
