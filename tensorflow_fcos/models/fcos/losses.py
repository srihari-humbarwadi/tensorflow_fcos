import tensorflow as tf


@tf.function
def focal_loss(y_true,
               y_pred,
               normalizer_value,
               num_classes=10,
               alpha=0.25,
               gamma=2):
    y_true = tf.one_hot(
        tf.cast(y_true, dtype=tf.int32), depth=num_classes + 1)
    y_true = y_true[:, :, 1:]
    y_pred_ = tf.sigmoid(y_pred)

    at = alpha * y_true + (1 - y_true) * (1 - alpha)
    pt = y_true * y_pred_ + (1 - y_true) * (1 - y_pred_)
    f_loss = at * \
        tf.pow(1 - pt, gamma) * \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
    f_loss = tf.reduce_sum(f_loss, axis=2)
    f_loss = tf.reduce_sum(f_loss, axis=1, keepdims=True)
    f_loss = f_loss / normalizer_value
    return f_loss


@tf.function
def centerness_loss(labels, logits, fg_mask, normalizer_value):
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    bce_loss = bce_loss * tf.expand_dims(fg_mask, axis=-1)
    bce_loss = tf.reduce_sum(bce_loss, axis=1)
    bce_loss = bce_loss / normalizer_value
    return bce_loss


@tf.function
def iou_loss(labels, logits, centers, fg_mask, normalizer_value):
    boxes_true = tf.concat([
        centers - labels[:, :, :2],
        centers + labels[:, :, 2:]], axis=-1)

    boxes_pred = tf.concat([
        centers - logits[:, :, :2],
        centers + logits[:, :, 2:]], axis=-1)

    lu = tf.maximum(boxes_true[:, :, :2], boxes_pred[:, :, :2])
    rd = tf.minimum(boxes_true[:, :, 2:], boxes_pred[:, :, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes_true_area = tf.reduce_prod(
        boxes_true[:, :, 2:] - boxes_true[:, :, :2], axis=2)
    boxes_pred_area = tf.reduce_prod(
        boxes_pred[:, :, 2:] - boxes_pred[:, :, :2], axis=2)
    union_area = tf.maximum(
        boxes_true_area + boxes_pred_area - intersection_area, 1e-8)
    iou = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    bg_mask = (1 - fg_mask) * 1e-8
    iou_loss = iou + bg_mask
    iou_loss = -1 * tf.math.log(iou_loss)
    iou_loss = iou_loss * fg_mask
    iou_loss = tf.reduce_sum(iou_loss, axis=1, keepdims=True)
    iou_loss = iou_loss / normalizer_value
    return iou_loss
