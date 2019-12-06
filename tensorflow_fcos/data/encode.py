import tensorflow as tf
from data.tf_record_parser import parse_example


@tf.function
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


@tf.function
def random_jitter(image):
    # ToDo
    pass


def compute_area(boxes):
    h_ = boxes[:, 2] - boxes[:, 0]
    w_ = boxes[:, 3] - boxes[:, 1]
    return h_ * w_


def compute_feature_sizes(H, W):
    fm_sizes = []
    for i in range(3, 8):
        stride = 2.**i
        fm_sizes.append([tf.math.ceil(H / stride),
                         tf.math.ceil(W / stride), stride])
    return fm_sizes


def get_centers(fm_h, fm_w, stride=None):
    rx = (tf.range(fm_w) + 0.5) * (stride)
    ry = (tf.range(fm_h) + 0.5) * (stride)
    sx, sy = tf.meshgrid(rx, ry)
    cxy = tf.stack([sx, sy], axis=-1)
    return cxy


def get_all_centers(H, W):
    centers_list = []
    feature_sizes = compute_feature_sizes(H, W)
    for fm_h, fm_w, stride in feature_sizes:
        cyx = get_centers(fm_h, fm_w, stride)
        cyx = tf.reshape(cyx, shape=[-1, 2])
        centers_list.append(cyx)
    return centers_list


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32)
])
def compute_targets_(centers, labels, low, high):
    '''
        From the FCOS paper, "Specifically, location (x, y) is
        considered as a positive sample if it falls into any
        ground-truth box and the class label c* of the location is
        the class label of the ground-truth box. Otherwise it is a
        negative sample and class* = 0 (background class)
        Besides the label for classification, we also have a 4D
        real vector t* = (l*, t*, r*, b*) being the regression
        targets for the location. Here l*, t*, r* and b* are the
        distances from the location to the four sides of the bounding
        box ...If a location falls into multiple bounding boxes, it is
        considered as an ambiguous sample. We simply choose the
        bounding box with minimal area as its regression target.
        ...we firstly compute the regressiontargets l*, t*, r* and b*
        for each location on all feature levels. Next, if a location
        satisfies max(l*, t*, r*, b*) > mi or max(l*, t*, r*, b*) < mi−1,
        it is set as a negative sample and is thus not required to
        regress a bounding box anymore. Here mi is the maximum distance
        that feature level i needs to regress. In this work, m2, m3, m4,
        m5, m6 and m7 are set as 0, 64, 128, 256, 512 and ∞, respectively"
        Args:
            centers (M, 2): Centers for the current feature level
            labels (N, 5):  All labels for the current image
            low: Lower limit for ltrb value for the current feature level
            high: Upper limit for ltrb value for the current feature level
    '''
    boxes_ = labels[:, :4]
    class_ids_ = labels[:, 4]

    # Sorted the boxes by area in ascending order so that
    # we pick the smallest box when computing ltbr values
    areas = compute_area(boxes_)
    sorted_indices = tf.argsort(areas)
    boxes = tf.gather(boxes_, indices=sorted_indices)
    class_ids = tf.gather(class_ids_, indices=sorted_indices)

    xy_min_ = boxes[:, :2]
    xy_max_ = boxes[:, 2:]
    lt_ = centers[:, None] - xy_min_
    rb_ = xy_max_ - centers[:, None]
    ltrb_ = tf.concat([lt_, rb_], axis=2)  # (M, N, 4)

    # check if max(lbtr) lies in the valid_range for this
    # feature level
    max_ltrb_ = tf.reduce_max(ltrb_, axis=2)  # (M, N)
    mask_ltrb_size = tf.logical_and(max_ltrb_ > low, max_ltrb_ < high)

    mask_lt = tf.logical_and(ltrb_[:, :, 0] > 0, ltrb_[:, :, 1] > 0)
    mask_rb = tf.logical_and(ltrb_[:, :, 2] > 0, ltrb_[:, :, 3] > 0)
    mask = tf.logical_and(mask_lt, mask_rb)
    mask = tf.logical_and(mask, mask_ltrb_size)  # (M, N)

    mask = tf.cast(mask, dtype=tf.float32)
    fg_mask = tf.reduce_sum(mask, axis=1) != 0  # (M,)
    fg_mask = tf.cast(fg_mask, dtype=tf.float32)
    fg_mask = tf.tile(fg_mask[:, None], multiples=[1, 4])

    valid_indices = tf.argmax(mask, axis=1)  # (M, )
    matched_boxes = tf.gather(boxes, valid_indices)
    matched_class_ids = tf.gather(class_ids, valid_indices) + 1

    x_min, y_min, x_max, y_max = tf.split(matched_boxes,
                                          num_or_size_splits=4,
                                          axis=1)
    l = tf.abs(centers[:, 0] - x_min[:, 0])
    t = tf.abs(centers[:, 1] - y_min[:, 0])
    r = tf.abs(x_max[:, 0] - centers[:, 0])
    b = tf.abs(y_max[:, 0] - centers[:, 1])
    lr = tf.stack([l, r], axis=1)
    tb = tf.stack([t, b], axis=1)

    min_lr = tf.reduce_min(lr, axis=1)
    max_lr = tf.reduce_max(lr, axis=1)
    min_tb = tf.reduce_min(tb, axis=1)
    max_tb = tf.reduce_max(tb, axis=1)

    classification_target = matched_class_ids * fg_mask[:, 0]
    centerness_target = tf.sqrt(
        (min_lr / max_lr) * (min_tb / max_tb)) * fg_mask[:, 0]
    regression_target = tf.stack([l, t, r, b], axis=1) * fg_mask
    return classification_target, centerness_target, regression_target


def compute_targets(H, W, labels):
    centers_list = get_all_centers(H, W)
    m = [
        [0.0, 64.0],
        [64.0, 128.0],
        [128.0, 256.0],
        [256.0, 512.0],
        [512.0, 1e8]]
    classification_target = []
    centerness_target = []
    regression_target = []
    for i in range(5):
        centers = centers_list[i]
        low, high = m[i]
        cls_target, \
            ctr_target, \
            reg_target = compute_targets_(centers, labels, low, high)

        classification_target.append(cls_target)
        centerness_target.append(ctr_target)
        regression_target.append(reg_target)

    classification_target = tf.concat(classification_target, axis=0)
    classification_target = tf.expand_dims(classification_target, axis=-1)
    classification_target = tf.tile(classification_target,
                                    multiples=[1, 10])
    centerness_target = tf.concat(centerness_target, axis=0)
    regression_target = tf.concat(regression_target, axis=0)
    return classification_target, centerness_target, regression_target


def load_data(h, w):
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
        image, boxes = flip_data(image, boxes, w)
        label = tf.concat([boxes, class_ids], axis=-1)
        classification_target, centerness_target, regression_target = \
            compute_targets(h, w, label)
        return image, \
            (classification_target, centerness_target, regression_target)
    return load_data_
