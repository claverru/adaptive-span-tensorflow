import tensorflow as tf


class AdaptiveMask(tf.keras.layers.Layer):
	"""docstring for AdaptiveMask"""
	def __init__(self, max_size, ramp_size, init_val=0., shape=(1,)):
		super(AdaptiveMask, self).__init__()
		self._max_size = max_size
		self._ramp_size = ramp_size
		self.init_val = init_val
		self.shape = shape
		self.mask_template = tf.linspace(1. - max_size, 0., num=max_size)
		# self.register_buffer('mask_template', mask_template)

	def build(self, input_shape):
		self.current_val = tf.Variable(
				tf.fill(dims=self.shape, value=self.init_val))
		super(AdaptiveMask, self).build(input_shape)

	def get_current_max_size(self, include_ramp=True):
		current_size = tf.math.ceil(
			tf.reduce_max(self.current_val).numpy() * self._max_size)
		if include_ramp:
			current_size += self._ramp_size
		current_size = tf.maximum(0, min(self._max_size, current_size)).numpy()
		return current_size

	def get_current_avg_size(self, include_ramp=True):
		current_size = tf.math.ceil(
			tf.reduce_mean(self.current_val).numpy() * self._max_size)
		if include_ramp:
			current_size += self._ramp_size
		current_size = tf.maximum(0, min(self._max_size, current_size)).numpy()
		return current_size

	# MinMaxNorm Keras constraint maybe?!!!!!
	def clamp_param(self):
		"""this need to be called after each update"""
		self.current_val.data = tf.clip_by_value(
				self.current_val, 0, 1)

	def call(self, x):
		mask = self.mask_template + self.current_val * self._max_size
		mask = mask / self._ramp_size + 1
		mask = tf.clip_by_value(mask, 0, 1)
		if x.shape[-1] < self._max_size:
			# the input could have been trimmed beforehand to save computation
			mask = mask[:, :, -x.shape[-1]:]
		x = x * mask
		return x


class AdaptiveSpan(tf.keras.layers.Layer):
	"""Adaptive attention span for Transformerself.
	This module learns an attention span length from data for each
	self-attention head.
	Args:
	    attn_span: maximum attention span
	    adapt_span_loss: loss coefficient for the span length
	    adapt_span_ramp: length of the masking ramp
	    adapt_span_init: initial size ratio
	    adapt_span_cache: adapt cache size to reduce memory usage
	"""
	def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
				 adapt_span_init, adapt_span_cache, nb_heads, **kargs):
		super(AdaptiveSpan, self).__init__()
		self._adapt_cache = adapt_span_cache
		self._max_span = attn_span
		self._loss_coeff = adapt_span_loss
		self._nb_heads = nb_heads
		self._adapt_span_ramp = adapt_span_ramp
		self._adapt_span_init = adapt_span_init

	def build(self, input_shape):
		self._mask = AdaptiveMask(
				max_size=self._max_span,
				ramp_size=self._adapt_span_ramp,
				init_val=self._adapt_span_init,
				shape=(self._nb_heads, 1, 1))

