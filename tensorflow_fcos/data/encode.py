import tensorflow as tf
from data.tf_record_parser import parse_example

all_centers = [None] * 8


def flip_data(image, boxes, w):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([
            w - boxes[:, 2],
            boxes[:, 1],
            w - boxes[:, 0],
            boxes[:, 3]
        ], axis=-1)
    return image, boxes


def random_jitter(image):
    return image


def compute_area(boxes):
    h_ = boxes[:, 2] - boxes[:, 0]
    w_ = boxes[:, 3] - boxes[:, 1]
    return h_ * w_


def get_centers(level, h, w):
    stride = tf.cast(tf.pow(2, level), dtype=tf.float32)
    fm_h = tf.math.ceil(h / stride)
    fm_w = tf.math.ceil(w / stride)
    rx = (tf.range(fm_w, dtype=tf.float32) + 0.5) * (stride)
    ry = (tf.range(fm_h, dtype=tf.float32) + 0.5) * (stride)
    sx, sy = tf.meshgrid(rx, ry)
    cxy = tf.stack([sx, sy], axis=-1)
    cxy = tf.reshape(cxy, shape=[-1, 2])
    return cxy


def generate_centers(h, w):
    global all_centers
    for level in range(3, 8):
        all_centers[level] = get_centers(level, h, w)


def compute_target_(level, boxes, class_ids, low, high, h, w):
    centers = all_centers[level]

    xy_min_ = boxes[:, :2]
    xy_max_ = boxes[:, 2:]
    lt_ = tf.expand_dims(centers, axis=1) - xy_min_
    rb_ = xy_max_ - tf.expand_dims(centers, axis=1)
    ltrb_ = tf.concat([lt_, rb_], axis=2)

    max_ltrb_ = tf.reduce_max(ltrb_, axis=2)
    mask_size = tf.logical_and(tf.greater(max_ltrb_, low),
                               tf.less(max_ltrb_, high))

    mask = tf.cast(tf.greater(ltrb_, 0), dtype=tf.float32)
    mask = tf.not_equal(tf.reduce_prod(mask, axis=2), 0.)
    mask = tf.logical_and(mask, mask_size)
    mask = tf.cast(mask, dtype=tf.float32)

    fg_mask = tf.not_equal(tf.reduce_sum(mask, axis=1), 0)
    fg_mask = tf.cast(fg_mask, dtype=tf.float32)

    valid_indices = tf.argmax(mask, axis=1)
    matched_boxes = tf.gather(boxes, valid_indices)
    matched_class_ids = tf.gather(class_ids, valid_indices) + 1

    x_min, y_min, x_max, y_max = tf.split(matched_boxes,
                                          num_or_size_splits=4,
                                          axis=1)
    _l = tf.abs(centers[:, 0] - x_min[:, 0])
    t = tf.abs(centers[:, 1] - y_min[:, 0])
    r = tf.abs(x_max[:, 0] - centers[:, 0])
    b = tf.abs(y_max[:, 0] - centers[:, 1])
    lr = tf.stack([_l, r], axis=1)
    tb = tf.stack([t, b], axis=1)

    min_lr = tf.reduce_min(lr, axis=1)
    max_lr = tf.reduce_max(lr, axis=1)
    min_tb = tf.reduce_min(tb, axis=1)
    max_tb = tf.reduce_max(tb, axis=1)

    classification_target = matched_class_ids * fg_mask
    centerness_target = tf.sqrt(
        (min_lr / max_lr) * (min_tb / max_tb)) * fg_mask
    centerness_target = tf.reshape(centerness_target, shape=[-1, 1])
    regression_target = tf.stack(
        [_l, t, r, b], axis=1) * tf.expand_dims(fg_mask, axis=1)

    return (classification_target,
            centerness_target,
            regression_target,
            fg_mask)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32)])
def compute_targets(labels, h, w):
    boxes_ = labels[:, :4]
    class_ids_ = labels[:, 4]

    areas = compute_area(boxes_)
    sorted_indices = tf.argsort(areas, direction='ASCENDING')
    boxes = tf.gather(boxes_, indices=sorted_indices)
    class_ids = tf.gather(class_ids_, indices=sorted_indices)

    P3_target = compute_target_(3, boxes, class_ids, 0, 64, h, w)
    P4_target = compute_target_(4, boxes, class_ids, 64, 128, h, w)
    P5_target = compute_target_(5, boxes, class_ids, 128, 256, h, w)
    P6_target = compute_target_(6, boxes, class_ids, 256, 512, h, w)
    P7_target = compute_target_(7, boxes, class_ids, 512, 1e8, h, w)

    classification_target = tf.concat([P3_target[0],
                                       P4_target[0],
                                       P5_target[0],
                                       P6_target[0],
                                       P7_target[0]], axis=0)

    centerness_target = tf.concat([P3_target[1],
                                   P4_target[1],
                                   P5_target[1],
                                   P6_target[1],
                                   P7_target[1]], axis=0)

    regression_target = tf.concat([P3_target[2],
                                   P4_target[2],
                                   P5_target[2],
                                   P6_target[2],
                                   P7_target[2]], axis=0)

    fg_mask = tf.concat([P3_target[3],
                         P4_target[3],
                         P5_target[3],
                         P6_target[3],
                         P7_target[3]], axis=0)

    normalizer_value = tf.maximum(tf.reduce_sum(fg_mask, keepdims=True), 1.0)

    return (classification_target,
            centerness_target,
            regression_target,
            fg_mask,
            normalizer_value)


def load_data(h, w):
    generate_centers(h, w)

    @tf.function
    def load_data_(example_proto):
        image, boxes_, class_ids = parse_example(example_proto)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, size=[h, w])
        boxes = tf.stack([
            tf.clip_by_value(boxes_[:, 0] * w, 0, w),
            tf.clip_by_value(boxes_[:, 1] * h, 0, h),
            tf.clip_by_value(boxes_[:, 2] * w, 0, w),
            tf.clip_by_value(boxes_[:, 3] * h, 0, h)
        ], axis=-1)
        image = image / 255.
        labels = tf.concat([boxes, class_ids], axis=-1)
        targets = compute_targets(labels, h, w)
        return image, targets
    return load_data_
