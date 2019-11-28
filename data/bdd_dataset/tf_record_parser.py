import tensorflow as tf

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'xmins': tf.io.VarLenFeature(tf.float32),
    'ymins': tf.io.VarLenFeature(tf.float32),
    'xmaxs': tf.io.VarLenFeature(tf.float32),
    'ymaxs': tf.io.VarLenFeature(tf.float32),
    'labels': tf.io.VarLenFeature(tf.float32)
}


def parse_example(H, W):
	@tf.function
	def parse_example_(example_proto):
	    parsed_example = tf.io.parse_single_example(
	        example_proto, feature_description)
	    image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
	    bboxes = tf.stack([
	        tf.sparse.to_dense(parsed_example['xmins']) * W,
	        tf.sparse.to_dense(parsed_example['ymins']) * H,
	        tf.sparse.to_dense(parsed_example['xmaxs']) * W,
	        tf.sparse.to_dense(parsed_example['ymaxs']) * H
	    ], axis=-1)
	    class_ids = tf.reshape(tf.sparse.to_dense(
	        parsed_example['labels']), [-1, 1])
	    return image, bboxes, class_ids
	return parse_example_